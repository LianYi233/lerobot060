"""CPU unit tests; synthetic inputs only, not a test of the trained checkpoint."""
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

SCRIPT = Path(__file__).with_name('plot_task2_teleoperation_vs_model.py')
spec = importlib.util.spec_from_file_location('task2_plot', SCRIPT)
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)


def stub_policy(path):
    path.mkdir(parents=True)
    (path / 'config.json').write_text('{}')
    (path / 'model.safetensors').write_bytes(b'test-only')
    return path


def test_resolve_requested_step_not_latest(tmp_path):
    group = tmp_path / 'group'
    p = stub_policy(group / 'run/checkpoints/020000/pretrained_model')
    stub_policy(group / 'run/checkpoints/010000/pretrained_model')
    assert m.resolve_policy(group, 20000) == p
    assert m.resolve_policy(p, 20000) == p


def test_resolve_ambiguous_fails(tmp_path):
    stub_policy(tmp_path / 'a/checkpoints/020000/pretrained_model')
    stub_policy(tmp_path / 'b/checkpoints/020000/pretrained_model')
    with pytest.raises(ValueError, match='found 2'):
        m.resolve_policy(tmp_path, 20000)


def test_saved_split_and_even_selection(tmp_path):
    path = tmp_path / 'split.json'
    path.write_text(json.dumps({'train_episodes': list(range(80)), 'eval_episodes': list(range(80, 90))}))
    groups, origin = m.partition_episodes(SimpleNamespace(total_episodes=90), {}, path)
    episodes, split = m.select_episodes('val', groups, 3)
    assert episodes == [80, 84, 89]
    assert split == 'val'
    assert origin == str(path)


def test_split_fallback_matches_tail_by_task():
    metadata = SimpleNamespace(total_episodes=10, episodes=[{'tasks': ['one']} for _ in range(10)])
    groups, _ = m.partition_episodes(metadata, {'eval_split': 0.2}, None)
    assert groups['val'] == [8, 9]
    assert groups['train'] == list(range(8))


def test_missing_split_not_guessed():
    with pytest.raises(ValueError, match='will not guess'):
        m.partition_episodes(SimpleNamespace(total_episodes=10), {}, None)


def test_overlapping_split_fails(tmp_path):
    path = tmp_path / 'split.json'
    path.write_text(json.dumps({'train_episodes': [0, 1], 'eval_episodes': [1]}))
    with pytest.raises(ValueError, match='overlap'):
        m.partition_episodes(SimpleNamespace(total_episodes=2), {}, path)


def test_episode_indices_are_relative_not_absolute():
    rows = [{'episode_index': ep, 'frame_index': f, 'index': start + f}
            for ep, start, size in [(83, 4000, 10), (87, 9000, 9)] for f in range(size)]
    anchors, per_ep = m.index_episodes(rows, [83, 87], 8)
    assert [(r['episode'], r['frame'], r['relative_index']) for r in anchors] == [
        (83, 0, 0), (83, 8, 8), (87, 0, 10), (87, 8, 18)]
    assert len(per_ep[87]) == 9


def test_missing_frame_fails():
    rows = [{'episode_index': 83, 'frame_index': 1, 'index': 1001}]
    with pytest.raises(ValueError, match='incomplete'):
        m.index_episodes(rows, [83], 8)


def target_item():
    return {'action': np.zeros((50, 7), np.float32),
            'action_is_pad': np.arange(50) >= 3, 'index': 107, 'frame_index': 7}


def test_tail_padding_excluded():
    target, valid = m.recorded_targets(target_item(), {'dataset_from_index': 100, 'dataset_to_index': 110}, 50, 7)
    assert valid.sum() == 3
    assert target.shape == (50, 7)


def test_invalid_padding_fails():
    item = target_item()
    item['action_is_pad'][3] = False
    with pytest.raises(ValueError, match='boundary'):
        m.recorded_targets(item, {'dataset_from_index': 100, 'dataset_to_index': 110}, 50, 7)


def synthetic_chunks(length=19):
    frames = np.arange(0, length, 8)
    result = {key: [] for key in ('predicted','target','hold','valid')}
    for start in frames:
        offsets = np.arange(50)
        target = np.repeat((start + offsets)[:, None], 7, axis=1).astype(np.float32) * 0.01
        prediction = target + .02
        hold = np.full((50,7), start*.01, np.float32)
        for key, val in [('target', target), ('predicted', prediction), ('hold', hold), ('valid', start+offsets < length)]:
            result[key].append(val)
    result = {key:np.asarray(value) for key,value in result.items()}
    result.update(episode_index=np.full(len(frames),83), frame_index=frames)
    return result


def test_stitch_has_no_duplicate_or_missing_frames():
    values = synthetic_chunks()
    joined = m.stitch_episode(values, 83, 8, 19)
    np.testing.assert_allclose(joined['target'][:,0], np.arange(19)*.01)
    np.testing.assert_allclose(joined['predicted'][:,0], np.arange(19)*.01+.02, atol=1e-8)
    assert joined['anchor_frame'].tolist() == [0]*8+[8]*8+[16]*3


def test_stitch_detects_overlap():
    values = synthetic_chunks()
    values['frame_index'][1] = 7
    with pytest.raises(ValueError, match='overwrite'):
        m.stitch_episode(values, 83, 8, 19)


def test_stitch_detects_missing_coverage():
    values = synthetic_chunks()
    values = {k:v[:2] for k,v in values.items()}
    with pytest.raises(ValueError, match='Incomplete'):
        m.stitch_episode(values, 83, 8, 19)


def test_metrics_h8_does_not_count_padding():
    values = synthetic_chunks()
    scores = m.metrics(values, m.NAMES, [1,8,50], 1.)
    assert scores['8']['valid_anchor_offset_pairs'] == 19
    assert scores['50']['valid_anchor_offset_pairs'] == 19+11+3
    assert scores['8']['joint_mae_deg'] == pytest.approx(np.rad2deg(.02), rel=1e-6)
    assert scores['8']['gripper_mae_dataset_units'] == pytest.approx(.02, rel=1e-6)


def test_motion_metrics_match_known_values():
    target = np.ones((1,8,7)) * np.deg2rad(10)
    prediction = target * .8
    values = {'target': target, 'predicted': prediction, 'hold':np.zeros_like(target), 'valid':np.ones((1,8),bool)}
    motion = m.metrics(values, m.NAMES, [8],1.)['8']['per_dimension']['joint_2.pos']['moving_coordinate']
    assert motion['direction_agreement'] == 1.
    assert motion['amplitude_ratio'] == pytest.approx(.8)
    assert motion['delta_mae_deg'] == pytest.approx(2.)


def test_plot_outputs_exact_legend_labels(tmp_path):
    stem = tmp_path/'joint_2'
    m.draw_pair(stem, np.arange(8)/10, np.arange(8)*.1, np.arange(8)*.09,
                'joint_2.pos', 'SYNTHETIC UNIT TEST - NOT MODEL OUTPUT', 'Seconds', dpi=90)
    assert stem.with_suffix('.png').stat().st_size > 5000
    svg = stem.with_suffix('.svg').read_text()
    assert 'teleoperation' in svg and 'action from model' in svg


def test_parser_fixed_execution_horizon():
    with pytest.raises(SystemExit):
        m.parse_args(['--execution-steps','10'])
    args = m.parse_args([])
    assert args.checkpoint_step == 20000
    assert args.episodes == 'val'
    assert args.num_episodes == 3


def test_end_to_end_with_stub_policy_no_future_label_input(tmp_path, monkeypatch):
    """Exercise I/O and plotting with mock LeRobot interfaces, NOT the trained model."""
    import types
    import torch
    from safetensors.torch import save_file

    repo = tmp_path / 'repo'
    repo.mkdir()
    for filename in ('src/lerobot/policies/pi05/modeling_pi05.py',
                     'src/lerobot/policies/pi05/processor_pi05.py',
                     'src/lerobot/policies/pi_gemma.py'):
        path = repo / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('# test stub\n')
    data_root = tmp_path / m.TASK
    (data_root/'meta').mkdir(parents=True)
    info = {'codebase_version':'v3.0','fps':10}
    (data_root/'meta/info.json').write_text(json.dumps(info))
    policy_path = tmp_path / 'run/checkpoints/020000/pretrained_model'
    policy_path.mkdir(parents=True)
    (policy_path/'config.json').write_text('{}')
    (policy_path/'train_config.json').write_text(json.dumps({'dataset': {
        'root':str(data_root), 'repo_id':m.TASK_REPO,'eval_split':.25}}))
    tokenizer = tmp_path/'tokenizer'
    tokenizer.mkdir()
    (tokenizer/'tokenizer_config.json').write_text('{}')
    (tokenizer/'tokenizer.json').write_text('{}')
    weights = {'model.vlm_prompt_tokens.weight':torch.zeros(16,2),
               'model.prompt_tokens.weight':torch.zeros(16,2)}
    save_file(weights, str(policy_path/'model.safetensors'))

    class Config:
        training_stage='flow'; use_peft=False; chunk_size=50
        action_delta_indices=list(range(50)); n_action_steps=8
        use_relative_actions=False; piper_gripper_history_steps=0
        dtype='float32'; max_action_dim=32
        image_features={'observation.images.cam_high':None}

    class PreTrainedConfig:
        @classmethod
        def from_pretrained(cls, *a, **kw):
            assert kw['local_files_only']
            return Config()

    class Metadata:
        def __init__(self, *a, **kw):
            self.total_episodes=4
            self.episodes=[{'tasks':['remove the cuboid from blue plate'],
                            'dataset_from_index':ep*19,'dataset_to_index':(ep+1)*19}
                           for ep in range(4)]

    class Dataset:
        def __init__(self, *a, episodes, **kw):
            assert episodes == [3]
            assert kw['return_uint8'] and kw['image_transforms'] is None
        def select_columns(self, columns):
            return [{'episode_index':3,'frame_index':frame,'index':57+frame} for frame in range(19)]
        def get_raw_item(self, relative):
            return {'action':torch.full((7,),relative*.01)}
        def __getitem__(self, relative):
            future=torch.arange(relative,relative+50).clamp(max=18)
            return {'episode_index':3,'frame_index':relative,'index':57+relative,
                    'observation.images.cam_high':torch.zeros(3,16,16,dtype=torch.uint8),
                    'observation.state':torch.zeros(7),
                    'action':future.float()[:,None].expand(-1,7)*.01,
                    'action_is_pad':torch.arange(relative,relative+50)>=19,
                    'task':'remove the cuboid from blue plate'}

    class Processor:
        steps=[]
        def reset(self): pass
        def __call__(self, value): return value

    calls=[]
    class Policy:
        @classmethod
        def from_pretrained(cls, *a, **kw):
            assert kw['strict'] is True and kw['local_files_only'] is True
            return cls()
        def to(self,*a): return self
        def eval(self): return self
        def reset(self): pass
        def state_dict(self): return weights
        def predict_action_chunk(self, batch, noise):
            assert 'action' not in batch and 'action_is_pad' not in batch
            assert batch['observation.images.cam_high'].dtype == torch.float32
            assert noise.shape == (1,50,32)
            calls.append(1)
            return noise[:,:,:7]*.01

    def features(*a):
        return {'action':{'names':m.NAMES}, 'observation.state':{'names':m.NAMES},
                'observation.images.cam_high':{}}
    def module(name, **exports):
        obj=types.ModuleType(name)
        obj.__dict__.update(exports)
        monkeypatch.setitem(sys.modules,name,obj)
        return obj
    module('piper_policy_utils',check_checkpoint_files=lambda *a:None,
           check_prompt_weights=lambda *a:None,deployment_features=features)
    module('lerobot.configs.policies',PreTrainedConfig=PreTrainedConfig)
    module('lerobot.datasets.dataset_metadata',LeRobotDatasetMetadata=Metadata)
    module('lerobot.datasets.factory',resolve_delta_timestamps=lambda *a:{'action':list(range(50))})
    module('lerobot.datasets.lerobot_dataset',LeRobotDataset=Dataset)
    module('lerobot.policies.factory',make_pre_post_processors=lambda *a,**kw:(Processor(),Processor()))
    module('lerobot.policies.pi05.configuration_pi05',PI05Config=Config)
    module('lerobot.policies.pi05.modeling_pi05',PI05Policy=Policy)
    module('lerobot.policies.pi05',processor_pi05=object())
    monkeypatch.setattr(m,'activate_repo',lambda requested:repo)
    args=m.parse_args(['--policy-path',str(policy_path), '--dataset-root',str(data_root),
                       '--tokenizer-path',str(tokenizer), '--device','cpu',
                       '--num-episodes','1','--windows-per-episode','0',
                       '--output-dir',str(tmp_path/'output')])
    output=m.run(args)
    assert len(calls)==3
    summary=json.loads((output/'summary.json').read_text())
    assert summary['completed'] and summary['episodes']==[3]
    assert summary['horizons']['8']['valid_anchor_offset_pairs']==19
    assert len(list((output/'episode_000003/episode').glob('*.png')))==7
    assert (output/'index.html').is_file()
    with np.load(output/'predictions.npz',allow_pickle=False) as values:
        assert values['predicted'].shape==(3,50,7)

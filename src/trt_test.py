# 파일: 예) tools/viz_trt.py (또는 src/explore.py에 추가)
import os
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # noqa: F401
import torch
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.patches as mpatches

from src.data import compile_data
from src.tools import gen_dx_bx, get_nusc_maps, plot_nusc_map, add_ego, denormalize_img

class TRTEngineRunner:
    def __init__(self, engine_path):
        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()
        
        # 바인딩 정보 수집
        self.bindings = [None] * self.engine.num_bindings
        self.binding_names = []
        self.binding_is_input = []
        self.host_mem = []
        self.dev_mem = []
        for i in range(self.engine.num_bindings):
            name = self.engine.get_binding_name(i)
            self.binding_names.append(name)
            self.binding_is_input.append(self.engine.binding_is_input(i))
            dtype = trt.nptype(self.engine.get_binding_dtype(i))
            shape = tuple(self.context.get_binding_shape(i))
            size = int(np.prod(shape))
            h_mem = cuda.pagelocked_empty(size, dtype)
            d_mem = cuda.mem_alloc(h_mem.nbytes)
            self.host_mem.append(h_mem)
            self.dev_mem.append(d_mem)
            self.bindings[i] = int(d_mem)
            
        self.starter = torch.cuda.Event(enable_timing=True)
        self.ender = torch.cuda.Event(enable_timing=True)

    def infer(self, inputs_dict):
        # inputs_dict: {binding_name: np.ndarray (맞는 shape/dtype)}
        # 호스트 버퍼로 복사
        # for i, name in enumerate(self.binding_names):
        #     if self.binding_is_input[i]:
        #         arr = inputs_dict[name]
        #         assert arr.dtype == self.host_mem[i].dtype, f"dtype mismatch for {name}"
        #         assert arr.size == self.host_mem[i].size, f"size mismatch for {name}"
        #         np.copyto(self.host_mem[i], arr.ravel())
        #         cuda.memcpy_htod(self.dev_mem[i], self.host_mem[i])

        # # 실행
        # self.context.execute_v2(self.bindings)

        # # 출력 수집
        # outputs = {}
        # for i, name in enumerate(self.binding_names):
        #     if not self.binding_is_input[i]:
        #         cuda.memcpy_dtoh(self.host_mem[i], self.dev_mem[i])
        #         # 엔진 내 바인딩 shape 사용
        #         shape = tuple(self.context.get_binding_shape(i))
        #         outputs[name] = self.host_mem[i].reshape(shape)



        if self.engine.num_optimization_profiles > 0:
            try:
                self.context.set_optimization_profile_async(0, self.stream.handle)  # TRT 8.5+
            except AttributeError:
                self.context.active_optimization_profile = 0                   # 하위버전 호환

        # 2) 입력 바인딩 shape 지정
        for name, arr in inputs_dict.items():
            idx = self.engine.get_binding_index(name)
            shape = tuple(arr.shape)
            self.context.set_binding_shape(idx, shape)

        # 3) 모든 바인딩 shape 지정되었는지 확인
        assert self.context.all_binding_shapes_specified, "Binding shapes not fully specified"

        # 4) 지정된 shape로 호스트/디바이스 버퍼 준비(동적이면 매 실행마다 갱신 필요)
        bindings = [None] * self.engine.num_bindings
        host_mem, dev_mem = [], []
        for i in range(self.engine.num_bindings):
            dtype = trt.nptype(self.engine.get_binding_dtype(i))
            shape = tuple(self.context.get_binding_shape(i))  # now concrete
            size = int(np.prod(shape))
            h = cuda.pagelocked_empty(size, dtype)
            d = cuda.mem_alloc(h.nbytes)
            host_mem.append(h); dev_mem.append(d); bindings[i] = int(d)

        # 5) 입력 복사 → 실행 → 출력 복사
        for name, arr in inputs_dict.items():
            i = self.engine.get_binding_index(name)
            np.copyto(host_mem[i], arr.ravel())
            cuda.memcpy_htod(dev_mem[i], host_mem[i])

        self.starter.record()  # TRT 실행 시간 측정 시작
        self.context.execute_async_v2(bindings, self.stream.handle)  # 또는 execute_async_v2(stream)
        self.ender.record()  # TRT 실행 시간 측정 종료
        torch.cuda.synchronize()  # 동기화
        print("TRT 실행 시간:", self.starter.elapsed_time(self.ender), "ms")

        outputs = {}
        for i in range(self.engine.num_bindings):
            if not self.engine.binding_is_input(i):
                cuda.memcpy_dtoh(host_mem[i], dev_mem[i])
                shape = tuple(self.context.get_binding_shape(i))
                outputs[self.engine.get_binding_name(i)] = host_mem[i].reshape(shape)
        return outputs

def viz_model_preds_trt(version,
                        engine_path="liftsplatshoot.trt",
                        dataroot='data/nuScenes',
                        map_folder='data/nuScenes',
                        gpuid=0,
                        viz_train=False,
                        H=900, W=1600,
                        resize_lim=(0.193, 0.225),
                        final_dim=(128, 352),
                        bot_pct_lim=(0.0, 0.22),
                        rot_lim=(-5.4, 5.4),
                        rand_flip=True,
                        xbound=[-50.0, 50.0, 0.5],
                        ybound=[-50.0, 50.0, 0.5],
                        zbound=[-10.0, 10.0, 20.0],
                        dbound=[4.0, 45.0, 1.0],
                        bsz=1,
                        nworkers=4
                    ):
    grid_conf = {
        "xbound": xbound, 
        "ybound": ybound, 
        "zbound": zbound, 
        "dbound": dbound
    }
    cams = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
            'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
    
    data_aug_conf = {
                    'resize_lim': resize_lim,
                    'final_dim': final_dim,
                    'rot_lim': rot_lim,
                    'H': H, 'W': W,
                    'rand_flip': rand_flip,
                    'bot_pct_lim': bot_pct_lim,
                    'cams': cams,
                    'Ncams': 5,
                }
    
    trainloader, valloader = compile_data(version, dataroot, data_aug_conf=data_aug_conf,
                                          grid_conf=grid_conf, bsz=bsz, nworkers=nworkers,
                                          parser_name='segmentationdata')
    
    loader = trainloader if viz_train else valloader
    nusc_maps = get_nusc_maps(map_folder)

    # TRT 로더
    runner = TRTEngineRunner(engine_path)
    print("TRT bindings:", runner.binding_names)

    dx, bx, _ = gen_dx_bx(grid_conf['xbound'], grid_conf['ybound'], grid_conf['zbound'])
    dx, bx = dx[:2].numpy(), bx[:2].numpy()

    scene2map = {}
    for rec in loader.dataset.nusc.scene:
        log = loader.dataset.nusc.get('log', rec['log_token'])
        scene2map[rec['name']] = log['location']
    
    # 바인딩 이름 매핑(엔진에 맞춰 조정 필요)
    # 예: trtexec 로그에서 입력: x.1, rots, trans, post_trans / 출력: (이름 예: '1665')
    name_x = next(n for n in runner.binding_names if n.startswith("x"))
    name_rots = "rots"
    name_trans = "trans"
    name_intrins = "intrins"
    name_post_rots = "post_rots"
    name_post_trans = "post_trans"
    out_name = next(n for n in runner.binding_names if not runner.binding_is_input[runner.binding_names.index(n)])
    
    # 시각화 준비
    val = 0.01
    fH, fW = final_dim
    fig = plt.figure(figsize=(3*fW*val, (1.5*fW + 2*fH)*val))
    gs = mpl.gridspec.GridSpec(3, 3, height_ratios=(1.5*fW, fH, fH))
    gs.update(wspace=0.0, hspace=0.0, left=0.0, right=1.0, top=1.0, bottom=0.0)

    counter = 0
    with torch.no_grad():
        for batchi, (imgs, rots, trans, intrins, post_rots, post_trans, binimgs) in enumerate(loader):
            # TRT 입력 준비 (float32, C-Contiguous)
            # 엔진이 정적 배치/해상도 가정이라 bsz=1 권장
            assert imgs.shape[0] == 1, "TRT 엔진이 고정 배치라면 bsz=1로 실행하세요."
            x_np = imgs.cpu().numpy().astype(np.float32, copy=False)
            rots_np = rots.cpu().numpy().astype(np.float32, copy=False)
            trans_np = trans.cpu().numpy().astype(np.float32, copy=False)
            intrins_np = intrins.cpu().numpy().astype(np.float32, copy=False)
            post_rots_np = post_rots.cpu().numpy().astype(np.float32, copy=False)
            post_trans_np = post_trans.cpu().numpy().astype(np.float32, copy=False)

            # 엔진 바인딩 shape와 정확히 일치해야 함
            inputs = {
                name_x: x_np,
                name_rots: rots_np,
                name_trans: trans_np,
                name_intrins: intrins_np,
                name_post_rots: post_rots_np,
                name_post_trans: post_trans_np,
            }
            # runner.starter.record()
            outputs = runner.infer(inputs)
            # runner.ender.record()
            # torch.cuda.synchronize()
            # print(f"TRT inference time: {runner.starter.elapsed_time(runner.ender):.2f}ms")
            out = outputs[out_name]  # shape: [1, 1, 200, 200] 등
            out_t = torch.tensor(out).sigmoid().cpu()
            si = 0
            plt.clf()
            for imgi, img in enumerate(imgs[si]):
                ax = plt.subplot(gs[1 + imgi // 3, imgi % 3])
                showimg = denormalize_img(img)
                if imgi > 2:
                    showimg = showimg.transpose(Image.FLIP_LEFT_RIGHT)
                plt.imshow(showimg)
                plt.axis("off")
                plt.annotate(cams[imgi].replace("_", " "), (0.01, 0.92), xycoords="axes fraction")

            ax = plt.subplot(gs[0, :])
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])
            plt.setp(ax.spines.values(), color='b', linewidth=2)
            plt.legend(handles=[
                mpatches.Patch(color=(0.0, 0.0, 1.0, 1.0), label='Output Vehicle Segmentation'),
                mpatches.Patch(color='#76b900', label='Ego Vehicle'),
                mpatches.Patch(color=(1.00, 0.50, 0.31, 0.8), label='Map (for visualization purposes only)')
            ], loc=(0.01, 0.86))
            plt.imshow(out_t[si].squeeze(0), vmin=0, vmax=1, cmap='Blues')

            rec = loader.dataset.ixes[counter]
            plot_nusc_map(rec, nusc_maps, loader.dataset.nusc, scene2map, dx, bx)
            plt.xlim((out_t.shape[3], 0))
            plt.ylim((0, out_t.shape[3]))
            add_ego(bx, dx)

            imname = f'eval_trt{batchi:06}_{si:03}.jpg'
            print('saving', imname)
            plt.savefig(imname)
            counter += 1
            # 원한다면 몇 배치만 실행 후 break
            # if batchi > 10: break


def compare_trt_and_onnx(version,
                         engine_path="liftsplatshoot.trt",
                         onnx_path="liftsplatshoot.onnx",
                         dataroot='data/nuScenes',
                         gpuid=0,
                         viz_train=False,
                         H=900, W=1600,
                         resize_lim=(0.193, 0.225),
                         final_dim=(128, 352),
                         bot_pct_lim=(0.0, 0.22),
                         rot_lim=(-5.4, 5.4),
                         rand_flip=True,
                         xbound=[-50.0, 50.0, 0.5],
                         ybound=[-50.0, 50.0, 0.5],
                         zbound=[-10.0, 10.0, 20.0],
                         dbound=[4.0, 45.0, 1.0],
                         bsz=1,
                         nworkers=4,
                         save_images=True,
                         save_dir="results/compare"):
    """동일 배치로 ONNX Runtime와 TensorRT를 실행해 수치 비교."""
    try:
        import onnxruntime as ort
    except Exception as e:
        raise ImportError('onnxruntime가 필요합니다. pip install onnxruntime-gpu 또는 onnxruntime') from e

    # 데이터 설정
    grid_conf = {"xbound": xbound, "ybound": ybound, "zbound": zbound, "dbound": dbound}
    data_aug_conf = {
        'resize_lim': resize_lim,
        'final_dim': final_dim,
        'rot_lim': rot_lim,
        'H': H, 'W': W,
        'rand_flip': rand_flip,
        'bot_pct_lim': bot_pct_lim,
        'cams': ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'],
        'Ncams': 5,
    }
    trainloader, valloader = compile_data(version, dataroot, data_aug_conf=data_aug_conf,
                                          grid_conf=grid_conf, bsz=bsz, nworkers=nworkers,
                                          parser_name='segmentationdata')
    loader = trainloader if viz_train else valloader

    # ORT 세션
    providers = []
    if gpuid >= 0:
        try:
            providers = [('CUDAExecutionProvider', {'device_id': gpuid})]
        except Exception:
            providers = []
    providers.append('CPUExecutionProvider')
    sess = ort.InferenceSession(onnx_path, providers=providers)
    ort_in_names = [i.name for i in sess.get_inputs()]
    ort_out_names = [o.name for o in sess.get_outputs()]

    # TRT 러너
    runner = TRTEngineRunner(engine_path)
    trt_bindings = runner.binding_names

    # 바인딩 이름 유틸
    def pick_name(candidates, names):
        for c in candidates:
            if c in names:
                return c
        for n in names:
            for c in candidates:
                if c in n:
                    return n
        raise RuntimeError(f"바인딩 이름을 찾을 수 없음: {candidates}")

    name_x          = pick_name(["x", "x.1"], trt_bindings)
    name_rots       = pick_name(["rots"], trt_bindings)
    name_trans      = pick_name(["trans"], trt_bindings)
    name_intrins    = pick_name(["intrins", "intrin"], trt_bindings)
    name_post_rots  = pick_name(["post_rots", "postrots"], trt_bindings)
    name_post_trans = pick_name(["post_trans", "posttrans"], trt_bindings)
    trt_out_name    = next(n for n in trt_bindings if not runner.binding_is_input[runner.binding_names.index(n)])

    # 저장 디렉토리
    if save_images:
        os.makedirs(save_dir, exist_ok=True)

    # 한 배치만 비교
    for batchi, (imgs, rots, trans, intrins, post_rots, post_trans, binimgs) in enumerate(loader):
        assert imgs.shape[0] == bsz == 1, "compare는 bsz=1 가정"
        # numpy 변환
        x_np = np.ascontiguousarray(imgs.cpu().numpy().astype(np.float32, copy=False))
        rots_np = np.ascontiguousarray(rots.cpu().numpy().astype(np.float32, copy=False))
        trans_np = np.ascontiguousarray(trans.cpu().numpy().astype(np.float32, copy=False))
        intrins_np = np.ascontiguousarray(intrins.cpu().numpy().astype(np.float32, copy=False))
        post_rots_np = np.ascontiguousarray(post_rots.cpu().numpy().astype(np.float32, copy=False))
        post_trans_np = np.ascontiguousarray(post_trans.cpu().numpy().astype(np.float32, copy=False))

        # ORT feeds 매핑
        feeds = {}
        ort_map = {
            'x': x_np,
            'rots': rots_np,
            'trans': trans_np,
            'intrins': intrins_np,
            'post_rots': post_rots_np,
            'post_trans': post_trans_np,
        }
        
        np.savez('my_lss_inputs.npz', **ort_map)
        for nm in ort_in_names:
            if nm in ort_map:
                feeds[nm] = ort_map[nm]
            else:
                for k, v in ort_map.items():
                    if k in nm:
                        feeds[nm] = v
                        break

        # ORT 추론 (logits -> sigmoid)
        out_onnx = sess.run([ort_out_names[0]], feeds)[0]
        out_onnx = 1.0 / (1.0 + np.exp(-out_onnx))

        # TRT 추론 (logits -> sigmoid)
        trt_inputs = {
            name_x: x_np,
            name_rots: rots_np,
            name_trans: trans_np,
            name_intrins: intrins_np,
            name_post_rots: post_rots_np,
            name_post_trans: post_trans_np,
        }
        out_trt = runner.infer(trt_inputs)[trt_out_name]
        out_trt = 1.0 / (1.0 + np.exp(-out_trt))

        # 수치 비교
        mae = float(np.mean(np.abs(out_onnx - out_trt)))
        mx = float(np.max(np.abs(out_onnx - out_trt)))
        rel = float(np.mean(np.abs(out_onnx - out_trt) / (np.abs(out_onnx) + 1e-6)))
        corr = float(np.corrcoef(out_onnx.ravel(), out_trt.ravel())[0, 1])
        print(f"Compare ORT vs TRT -> MAE={mae:.6g}, MAX={mx:.6g}, REL={rel:.6g}, Corr={corr:.6f}")
        print(f"Sum ORT={out_onnx.sum():.6f}, TRT={out_trt.sum():.6f}")

        if save_images:
            import matplotlib.pyplot as plt
            si = 0
            fig, axs = plt.subplots(1, 3, figsize=(12, 4))
            axs[0].imshow(out_onnx[si].squeeze(0), vmin=0, vmax=1, cmap='Blues'); axs[0].set_title('ONNX')
            axs[1].imshow(out_trt[si].squeeze(0), vmin=0, vmax=1, cmap='Blues'); axs[1].set_title('TRT')
            diff = np.abs(out_onnx - out_trt)[si].squeeze(0)
            axs[2].imshow(diff, cmap='magma'); axs[2].set_title('|Diff|')
            for a in axs: a.axis('off')
            out_path = os.path.join(save_dir, f'compare_{batchi:06}.png')
            plt.tight_layout(); plt.savefig(out_path); plt.close(fig)
            print('saved', out_path)

        break
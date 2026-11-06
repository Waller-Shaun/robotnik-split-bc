"""[随机目标改动] 独立随机数生成器，不消耗传感器噪声/训练的全局随机数。"""
import numpy as np


TARGET_SAMPLING_VERSION = 'n_area_uniform_annulus_configurable_range_noncollinear_v2'
RADIUS_RANGE_M = (.8, 2.)
HEIGHT_RANGE_M = (.5, 1.3)
MIN_PAIR_XY_DISTANCE_M = .4
MIN_TRIANGLE_XY_AREA_M2 = .1


def sample_patrol_targets(seed, count=3, radius_range=RADIUS_RANGE_M,
                          height_range=HEIGHT_RANGE_M, min_spacing=MIN_PAIR_XY_DISTANCE_M):
    """全周水平圆环，按面积均匀抽样；连续高度，不筛选模型成功目标。

    [可调参数改动] 数量/半径/高度/间距可调；1～2点不检查三角形面积。
    3点以上要求至少存在包含第1点的面积>=0.1m²的三角形，避免整体近似共线。
    坐标以初始展示世界原点为中心。未按模型成功与否筛选目标。
    """
    radius_range=np.asarray(radius_range,dtype=float)
    height_range=np.asarray(height_range,dtype=float)
    if (not isinstance(count,(int,np.integer)) or isinstance(count,bool) or count<1
            or radius_range.shape!=(2,) or height_range.shape!=(2,)
            or not np.isfinite(radius_range).all() or not np.isfinite(height_range).all()
            or radius_range[0]<=0 or radius_range[1]<radius_range[0]
            or height_range[0]<.5 or height_range[1]>1.3 or height_range[1]<height_range[0]
            or not np.isfinite(min_spacing) or min_spacing<0):
        raise ValueError('Use count>=1, 0<radius_min<=radius_max, 0.5<=height_min<=height_max<=1.3, spacing>=0')
    if count>1 and min_spacing>2*radius_range[1]:
        raise ValueError('Target spacing exceeds the diameter of the sampling region')
    if count>=3 and 1.3*radius_range[1]**2<MIN_TRIANGLE_XY_AREA_M2:
        raise ValueError('Sampling region too small for the noncollinear target requirement')
    rng = np.random.default_rng(seed)
    for _ in range(10000):
        angles = rng.uniform(-np.pi, np.pi, count)
        radius = np.sqrt(rng.uniform(radius_range[0]**2, radius_range[1]**2, count))
        targets = np.column_stack([radius*np.cos(angles), radius*np.sin(angles),
                                   rng.uniform(*height_range, count)])
        xy = targets[:, :2]
        if count>1:
            distance=np.linalg.norm(xy[:,None,:]-xy[None,:,:],axis=2)
            if np.any(distance[np.triu_indices(count,1)]<min_spacing):continue
        if count>=3:
            vectors=xy[1:]-xy[0]
            areas=.5*np.abs(vectors[:,0,None]*vectors[None,:,1]-vectors[:,1,None]*vectors[None,:,0])
            if np.max(areas)<MIN_TRIANGLE_XY_AREA_M2:continue
        return targets
    raise ValueError('Could not place targets: reduce --num-targets / --min-target-spacing or widen --radius-range')


def sample_three_targets(seed):
    """兼容此前三点采样入口。"""
    return sample_patrol_targets(seed)

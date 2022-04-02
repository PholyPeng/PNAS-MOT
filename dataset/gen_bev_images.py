"""
to generate bird eye view, we have to filter point cloud
first. which means we have to limit coordinates


"""
import numpy as np
import cv2
# import mayavi.mlab as mlab


def load_pc(f):
    b = np.fromfile(f, dtype=np.float32)
    return b.reshape((-1, 4))[:, :3]

def pc_normalize(dt, epsilon = 1e-5):
    """
    dt:(n, 3)
    """
    mu = np.mean(dt,axis=0)
    return (dt-mu)
    # sigma = np.std(dt,axis=0)    
    # return (dt-mu) / (sigma + epsilon)

def gen_bev_map(pc, side_range=[-10, 10], fwd_range=[-20, 20], res=0.05):
    x = pc[:, 0]
    y = pc[:, 1]
    z = pc[:, 2]

    # filter point cloud
    f_filt = np.logical_and((x > fwd_range[0]), (x < fwd_range[1]))
    s_filt = np.logical_and((y > -side_range[1]), (y < -side_range[0]))
    filt = np.logical_and(f_filt, s_filt)
    indices = np.argwhere(filt).flatten()

    # keepers
    x = x[indices]
    y = y[indices]
    z = z[indices]

    # convert coordinates to
    x_img = (-y / res).astype(np.int32)
    y_img = (-x / res).astype(np.int32)
    # shifting image, make min pixel is 0,0
    x_img -= int(np.floor(side_range[0] / res))
    y_img += int(np.ceil(fwd_range[1] / res))

    # crop y to make it not bigger than 255
    height_range = (-2, 0.5)
    pixel_values = np.clip(a=z, a_min=height_range[0], a_max=height_range[1])

    def scale_to_255(a, min, max, dtype=np.uint8):
        return (((a - min) / float(max - min)) * 255).astype(dtype)

    pixel_values = scale_to_255(pixel_values,
                                min=height_range[0],
                                max=height_range[1])

    # according to width and height generate image
    w = 1 + int((side_range[1] - side_range[0]) / res)
    h = 1 + int((fwd_range[1] - fwd_range[0]) / res)
    im = np.zeros([h, w], dtype=np.uint8)
    im[y_img, x_img] = pixel_values
    #cropped_cloud = np.vstack([x, y, z]).transpose()
    #return im, cropped_cloud
    im = im[np.newaxis, ...] # 1 x 256 x 256
    return im


def generate_bev_from_lidar(pc, side_range= [-5, 5], fwd_range = [-5, 5], res = 0.0392):
    pc_norm = pc_normalize(pc)
    im = gen_bev_map(pc_norm, side_range, fwd_range, res)
    return im
    
    
if __name__ == '__main__':
    a = "./kitti_t_o/training/velodyne/0000/000000.bin"
    points = load_pc(a)
    res = 0.05
    # image size would be 400x800
    side_range = (-20, 20)
    fwd_range = (-20, 20)
    im, cropped_cloud = gen_bev_map(points, side_range, fwd_range, res)
    print(im.shape)
    cv2.imshow('rr', im)
    cv2.waitKey(0)
import pickle
import itertools
from operations import *


def measure_gpu_time(module, in_channels, resolution, batch_size, num_iterations):
    input = torch.rand(batch_size, in_channels, resolution, resolution)
    module.cuda()
    input = input.cuda()
    start_gpu = torch.cuda.Event(enable_timing=True)
    end_gpu = torch.cuda.Event(enable_timing=True)
    time_all = 0

    with torch.no_grad():
        for i in range(num_iterations):
            start_gpu.record()
            _ = module(input)
            end_gpu.record()
            torch.cuda.synchronize()
            time_all += start_gpu.elapsed_time(end_gpu) / 1000.0
    return time_all / (num_iterations * batch_size)


def generate_latency_lut(config_list, file_name, batch_size=1, num_iterations=400):
    latency_lut = {}
    op_names = OPS.keys()
    for op_name in op_names:
        latency_lut[op_name] = {}

    channels_list, resolution_list, stride_list, affine_list = config_list.values()

    for op_name in op_names:
        print(op_name)
        if op_name == 'conv_7x1_1x7':
            list_conf = itertools.product(channels_list, channels_list, channels_list, stride_list, affine_list, resolution_list)
            for in_channels, middle_channels, out_channels, stride, affine, resolution in list_conf:
                op = OPS['conv_7x1_1x7'](in_channels, middle_channels, out_channels, stride, affine)
                latency = measure_gpu_time(op, in_channels, resolution, batch_size, num_iterations)
                key = 'in_channels_' + str(in_channels) + '_middle_channels_' + str(middle_channels) + '_out_channels_' + str(out_channels) + '_stride_' + str(stride) + '_affine_' + str(affine)
                latency_lut[op_name][key] = latency
        elif op_name in ['none', 'avg_pool_3x3', 'max_pool_3x3']:
            list_conf = itertools.product(channels_list, stride_list, resolution_list)
            for in_channels, stride, resolution in list_conf:
                op = OPS[op_name](in_channels, stride, True)
                latency = measure_gpu_time(op, in_channels, resolution, batch_size, num_iterations)
                key = 'in_channels_' + str(in_channels) + '_out_channels_' + str(in_channels) + '_stride_' + str(stride)
                latency_lut[op_name][key] = latency
        else:
            list_conf = itertools.product(channels_list, channels_list, stride_list, affine_list, resolution_list)
            for in_channels, out_channels, stride, affine, resolution in list_conf:
                op = OPS[op_name](in_channels, out_channels, stride, affine)
                latency = measure_gpu_time(op, in_channels, resolution, batch_size, num_iterations)
                key = 'in_channels_' + str(in_channels) + '_out_channels_' + str(out_channels) + '_stride_' + str(stride) + '_affine_' + str(affine)
                latency_lut[op_name][key] = latency

    with open(file_name, 'wb') as f:
        pickle.dump(latency_lut, f)


config_list = {
    'channels_list': [64, 128, 256, 512, 1024],
    'resolution_list': [64, 128, 256],
    'stride_list': [1, 2],
    'affine_list': [True, False],
}

generate_latency_lut(config_list, 'latency_lut.pkl')

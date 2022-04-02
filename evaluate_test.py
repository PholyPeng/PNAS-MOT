from kitti_devkit.evaluate_tracking import evaluate
step = 1
result_path = "./experiments/resutls" 
part="val"
MOTA, MOTP, recall, prec, F1, fp, fn, id_switches = evaluate(str(step), result_path, part = part)
print(MOTA)
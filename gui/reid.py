import subprocess

def train(file_path, data_dir, train_csv_path, val_csv_path, name="ft_ResNet50", batchsize=32, total_epoch=60, 
          model="resnet_ibn", model_subtype="default", warm_epoch=0, save_freq=5, num_workers=2, lr=0.05,
          erasing_p=0.5, fp16=False, cosine=False, color_jitter=False, triplet=False, contrast=False,
          sphere=False, circle=False):
    command = [
        "python3", file_path,
        "--data_dir", data_dir,
        "--train_csv_path", train_csv_path,
        "--val_csv_path", val_csv_path,
        "--name", name,
        "--batchsize", str(batchsize),
        "--total_epoch", str(total_epoch),
        "--model", model,
        "--model_subtype", model_subtype,
        "--warm_epoch", str(warm_epoch),
        "--save_freq", str(save_freq),
        "--num_workers", str(num_workers),
        "--lr", str(lr),
        "--erasing_p", str(erasing_p),
    ]

    if fp16: command.append("--fp16")
    if cosine: command.append("--cosine")
    if color_jitter: command.append("--color_jitter")
    if triplet: command.append("--triplet")
    if contrast: command.append("--contrast")
    if sphere: command.append("--sphere")
    if circle: command.append("--circle")

    
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = process.communicate()
    # out = "\n".join(stdout.split("\n"))
    # err = "\n".join(stderr.split("\n"))

    return stdout, stderr

def test(file_path, data_dir, query_csv_path, gallery_csv_path, model_opts, checkpoint, batchsize=32, eval_gpu=False):
    command = [
        "python3", file_path,
        "--data_dir", data_dir,
        "--query_csv_path", query_csv_path,
        "--gallery_csv_path", gallery_csv_path,
        "--model_opts", model_opts,
        "--batchsize", str(batchsize),
        "--checkpoint", checkpoint,
    ]

    if eval_gpu: command.append("--eval_gpu")

    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = process.communicate()

    return stdout, stderr

import re
import os
import matplotlib.pyplot as plt

def clamp_loss_values(loss_values, threshold=5.0, upper=False):
    if upper:
        return [max(loss, threshold) for loss in loss_values]
    return [min(loss, threshold) for loss in loss_values]

def parse_log_file(log_file):
    psnr_values = []
    timestamps = []

    with open(log_file, 'r') as file:
        lines = file.readlines()
        t = 0

        for line in lines:
            if "Average PSNR" in line:
                psnr = re.findall(r'Average PSNR (\d+\.\d+) dB', line)
                if psnr:
                    psnr_values.append(float(psnr[0]))
                    timestamps.append(t*10)
                    t += 1

    return timestamps, psnr_values

def extract_pixel_from_log(log_file):
    pixel_loss = []
    timestamps = []

    with open(log_file, 'r') as file:
        lines = file.readlines()
        t = 0

        for line in lines:
            if "Loss content" in line:
                loss_content = re.findall(r'Loss content:\s+(\d+\.\d+)', line)
                if loss_content:
                    pixel_loss.append(float(loss_content[0]))
                    timestamps.append(t)
                    t += 1

    return timestamps, pixel_loss

def extract_fft_from_log(log_file):
    fft_loss = []
    timestamps = []

    with open(log_file, 'r') as file:
        lines = file.readlines()
        t = 0

        for line in lines:
            if "Loss content" in line:
                loss_fft = re.findall(r'Loss fft:\s+(\d+\.\d+)', line)
                if loss_fft:
                    fft_loss.append(float(loss_fft[0]))
                    timestamps.append(t)
                    t += 1

    return timestamps, fft_loss

def plot_multiple_psnr_curves(log_files):
    plt.figure(figsize=(10, 6))

    for log_file in log_files:
        filename = os.path.basename(log_file)
        filename = os.path.splitext(filename)[0]  # Remove file extension
        timestamps, psnr_values = parse_log_file(log_file)
        psnr_values = clamp_loss_values(psnr_values, threshold=30, upper=True)
        plt.plot(timestamps, psnr_values, marker='o', linestyle='-', label=filename, markersize=2, linewidth=1)

        # Display the last PSNR value
        last_psnr = psnr_values[-1]
        last_timestamp = timestamps[-1]
        plt.text(last_timestamp, last_psnr, f' {last_psnr:.2f}', fontsize=8, verticalalignment='bottom')

    plt.title('PSNR Curves from Multiple Log Files')
    plt.xlabel('EPOCH')
    plt.ylabel('PSNR (dB)')
    plt.legend()
    plt.grid(True)

def plot_multiple_pixel_loss_curves(log_files):
    plt.figure(figsize=(10, 6))

    for log_file in log_files:
        filename = os.path.basename(log_file)
        filename = os.path.splitext(filename)[0]  # Remove file extension
        timestamps, pixel_loss = extract_pixel_from_log(log_file)
        pixel_loss = clamp_loss_values(pixel_loss, threshold=0.05)

        timestamps = timestamps[::200]
        pixel_loss = pixel_loss[::200]

        plt.plot(timestamps, pixel_loss, marker='o', linestyle='-', label=filename, markersize=0.1, linewidth=1)

    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Pixel Loss Over Steps')
    plt.legend()
    plt.grid(True)

def plot_multiple_fft_loss_curves(log_files):
    plt.figure(figsize=(10, 6))

    for log_file in log_files:
        filename = os.path.basename(log_file)
        filename = os.path.splitext(filename)[0]  # Remove file extension
        timestamps, fft_loss = extract_fft_from_log(log_file)
        fft_loss = clamp_loss_values(fft_loss, threshold=1.5)

        timestamps = timestamps[::200]
        fft_loss = fft_loss[::200]

        plt.plot(timestamps, fft_loss, marker='o', linestyle='-', label=filename, markersize=0.1, linewidth=1)

    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('FFT Loss Over Steps')
    plt.legend()
    plt.grid(True)

if __name__ == '__main__':
    log_files = [
        '/home/cc/Documents/20.03.setup/tmp/mlp4.log',
        '/home/cc/Documents/20.03.setup/tmp/mlp2.log',
        '/home/cc/Documents/20.03.setup/tmp/mlp1.log',
        '/home/cc/Documents/20.03.setup/tmp/mlp0.log',
        '/home/cc/Documents/20.03.setup/ps_g4t_stopped.log', 
        '/home/cc/Documents/20.03.setup/ps_gl84t_stopped.log',
        '/home/cc/Documents/20.03.setup/ps_g4_final.log',
        '/home/cc/Documents/20.03.setup/ps_gl84_final.log',
        '/home/cc/Documents/20.03.setup/g4_final.log',
        '/home/cc/Documents/20.03.setup/g2_final.log',
        '/home/cc/Documents/20.03.setup/gl42_final.log',
        '/home/cc/Documents/20.03.setup/gl44_final.log',
        '/home/cc/Documents/20.03.setup/gl84_final.log',
        ]  
    plot_multiple_psnr_curves(log_files)
    plot_multiple_pixel_loss_curves(log_files)
    plot_multiple_fft_loss_curves(log_files)

    plt.show()
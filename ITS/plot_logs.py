import re
import os
import matplotlib.pyplot as plt

# Function to parse the log file and extract PSNR values and timestamps
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
    
    print("Timestamps:", timestamps)
    print("PSNR Values:", psnr_values)

    return timestamps, psnr_values

# Main function to plot PSNR curve
def plot_psnr_curve(log_file):
    timestamps, psnr_values = parse_log_file(log_file)

    plt.figure(figsize=(10, 6))
    plt.plot(timestamps, psnr_values, marker='o', linestyle='-',markersize=3)
    plt.title('PSNR Curve over Time')
    plt.xlabel('Time')
    plt.ylabel('PSNR (dB)')
    plt.grid(True)
    plt.show()


# Main function to plot PSNR curves for multiple log files
def plot_multiple_psnr_curves(log_files):
    plt.figure(figsize=(10, 6))

    for log_file in log_files:
        filename = os.path.basename(log_file)
        filename = os.path.splitext(filename)[0]  # Remove file extension
        timestamps, psnr_values = parse_log_file(log_file)
        plt.plot(timestamps, psnr_values, marker='o', linestyle='-', label=filename,markersize=3)

        # Display the last PSNR value
        last_psnr = psnr_values[-1]
        last_timestamp = timestamps[-1]
        plt.text(last_timestamp, last_psnr, f' {last_psnr:.2f}', fontsize=8, verticalalignment='bottom')

    plt.title('PSNR Curves from Multiple Log Files')
    plt.xlabel('EPOCH')
    plt.ylabel('PSNR (dB)')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    log_files = [
        '/home/cc/Documents/20.03.setup/g4_final.log',
        '/home/cc/Documents/20.03.setup/g2_final.log',
        '/home/cc/Documents/20.03.setup/ps_g4_final.log',
        '/home/cc/Documents/20.03.setup/tmp/ps_g4t.log', 
        '/home/cc/Documents/20.03.setup/tmp/ps_gl84.log',
        '/home/cc/Documents/20.03.setup/tmp/ps_gl84t.log',
        '/home/cc/Documents/20.03.setup/gl42_final.log',
        '/home/cc/Documents/20.03.setup/gl44_final.log',
        '/home/cc/Documents/20.03.setup/gl84_final.log',
        ]  
    plot_multiple_psnr_curves(log_files)

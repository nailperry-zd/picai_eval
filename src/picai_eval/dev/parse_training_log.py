# Define the function to find the minimum validation loss and its epoch
import re
def find_min_validation_loss(file_path):
    min_loss = float('inf')
    min_epoch = -1
    current_epoch = -1
    val_loss_map = {}

    with open(file_path, 'r') as file:
        for line in file:
            match_epoch = re.match(r'epoch:\s+(\d+)', line)
            if match_epoch:
                current_epoch = int(match_epoch.group(1))
                continue
            if 'validation loss:' in line:
                # Extract the validation loss
                # Use regex to find the number
                match = re.search(r'validation loss: (-?[0-9.]+)', line)
                if match:
                    loss = float(match.group(1))
                    val_loss_map[current_epoch] = loss
                    # Update minimum loss and corresponding epoch
                    if loss < min_loss:
                        min_loss = loss
                        min_epoch = current_epoch

    return min_epoch, min_loss, val_loss_map

if __name__ == "__main__":
    # Specify the path to your text file
    file_path = r"Y:\picai\workdir\nnUNet_results\nnUNet\3d_fullres\Task2402_Z_SSMNet\nnUNetTrainerV2_Loss_FL_and_CE_checkpoints_FL0__nnUNetPlansv2.1\fold_0\training_log_2025_6_17_12_11_30.txt"

    # Call the function and print the results
    epoch, loss, val_loss_map = find_min_validation_loss(file_path)
    print(f'Minimum validation loss: {loss} at epoch: {epoch}')
    roi = [749, 799, 849, 899, 949, 999]
    for i in roi:
        print(f'validation loss: {val_loss_map[i]} at epoch: {i}')
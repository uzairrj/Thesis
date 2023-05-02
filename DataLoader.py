import pandas as pd
from torch.utils.data import Dataset
from tqdm import tqdm
import math
import numpy as np

'''
Process the dataframe to generate the sequences of the pedistrian trajectories

parms:
dataframe -> Dataframe containing trajectories about the the pedistrians
obs_len -> length of the trajectories that needs to be observed for the prediction
pred_len -> length of the trajectories that needs to be predicted
threshold -> threshold for the non linear trajectories
min_ped -> minimum amount of the pedistrian in the trajectory
'''
class IndoorTrajectoryDataset(Dataset):
    def __init__(self, dataframe, obs_len=8, pred_len=12, skip=1, threshold=0.002, min_ped=1):
        dataframe = dataframe.sort_values(by="time")
        dataframe = dataframe.reset_index(drop=True)
        seq_len = obs_len + pred_len
        frames = dataframe.time.unique()
        frames_data = []
        num_peds_in_seq = []
        loss_mask_list = []
        seq_list = []
        seq_list_rel = []
        
        for key, value in tqdm(dataframe.groupby("time").groups.items()):
            frames_data.append(dataframe.iloc[value].values)
        
        num_sequences = int(math.ceil((len(frames) - seq_len + 1) / skip))

        for idx in tqdm(range(0, num_sequences * skip + 1, skip)):
            curr_seq_data = np.concatenate(frames_data[idx:idx + seq_len], axis=0)
            peds_in_curr_seq = np.unique(curr_seq_data[:, 0])
            curr_seq_rel = np.zeros((len(peds_in_curr_seq), 2, seq_len))
            curr_seq = np.zeros((len(peds_in_curr_seq), 2, seq_len))
            curr_loss_mask = np.zeros((len(peds_in_curr_seq), seq_len))

            num_peds_considered = 0

            _non_linear_ped = []
            for _, ped_id in enumerate(peds_in_curr_seq):
                curr_ped_seq = curr_seq_data[curr_seq_data[:, 0] == ped_id, :]
                # d = np.around(curr_ped_seq[:,2:4])
                curr_ped_seq[:,2:4] = np.around(curr_ped_seq[:,2:4].astype(float), decimals=4)
                pad_front = np.where(curr_ped_seq[0, 1] == frames)[0][0] - idx
                pad_end = np.where(curr_ped_seq[-1, 1] == frames)[0][0] - idx + 1
                if pad_end - pad_front != seq_len:
                    continue
                curr_ped_seq = np.transpose(curr_ped_seq[:, 2:4])
                curr_ped_seq = curr_ped_seq

                # Make coordinates relative
                rel_curr_ped_seq = np.zeros(curr_ped_seq.shape)
                rel_curr_ped_seq[:, 1:] = curr_ped_seq[:, 1:] - curr_ped_seq[:, :-1]

                _idx = num_peds_considered
                curr_seq[_idx, :, pad_front:pad_end] = curr_ped_seq
                curr_seq_rel[_idx, :, pad_front:pad_end] = rel_curr_ped_seq
                # Linear vs Non-Linear Trajectory
                # _non_linear_ped.append(
                #     poly_fit(curr_ped_seq, pred_len, threshold))
                curr_loss_mask[_idx, pad_front:pad_end] = 1
                num_peds_considered += 1

            if num_peds_considered > min_ped:
                #non_linear_ped += _non_linear_ped
                num_peds_in_seq.append(num_peds_considered)
                loss_mask_list.append(curr_loss_mask[:num_peds_considered])
                seq_list.append(curr_seq[:num_peds_considered])
                seq_list_rel.append(curr_seq_rel[:num_peds_considered])
    

        


if __name__ == "__main__":
    df = pd.read_csv("D:\\Thesis\\output\\German1\\tajectories_0.5mins.csv", sep=';')
    df = df.sort_values(by="time")
    obj = IndoorTrajectoryDataset(df[:10000].reset_index(drop=True))

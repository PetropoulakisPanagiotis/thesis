import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.font_manager as fm
import os
import numpy as np

parent_dir = './results_slam'

ate_rmse_all_scenes_list = []
ate_mean_all_scenes_list = []
ate_max_all_scenes_list = []

relative_trans_rmse_all_scenes_list = []
relative_trans_mean_all_scenes_list = []
relative_trans_max_all_scenes_list = []

relative_rot_rmse_all_scenes_list = []
relative_rot_mean_all_scenes_list = []
relative_rot_max_all_scenes_list = []

ate_rmse_std_all_scenes_list = []
ate_mean_std_all_scenes_list = []
ate_max_std_all_scenes_list = []

relative_trans_rmse_std_all_scenes_list = []
relative_trans_mean_std_all_scenes_list = []
relative_trans_max_std_all_scenes_list = []

relative_rot_rmse_std_all_scenes_list = []
relative_rot_mean_std_all_scenes_list = []
relative_rot_max_std_all_scenes_list = []

durations_list = []
durations_list_std = []

for folder in os.listdir(parent_dir):
    if os.path.isdir(os.path.join(parent_dir, folder)):
        if 'scene' in folder:
            scene_name = folder
            try: 
                ate_error_df = pd.read_csv(os.path.join(parent_dir, folder, 'ate_error.csv'))
                relative_error_df = pd.read_csv(os.path.join(parent_dir, folder, 'relative_error.csv'))
                durations_df = pd.read_csv(os.path.join(parent_dir, folder, 'durations.csv'))
            except:
                continue

            ate_error_std_df = pd.read_csv(os.path.join(parent_dir, folder, 'ate_error_std.csv'))
            relative_error_std_df = pd.read_csv(os.path.join(parent_dir, folder, 'relative_error_std.csv'))
            if 'abs_trans_rmse' not in ate_error_df.columns:
                continue
            
            if False:
                parent_dir_old = "./results_all"
                ate_error_old_df = pd.read_csv(os.path.join(parent_dir_old, folder, 'ate_error.csv'))
                relative_error_old_df = pd.read_csv(os.path.join(parent_dir_old, folder, 'relative_error.csv'))
               
                try:
                    ate_error_old_df = ate_error_old_df[ate_error_old_df['method'] != 'virtual-gt'] 
                    ate_error_old_df = ate_error_old_df[ate_error_old_df['method'] != 'global'] 
                    ate_error_old_df = ate_error_old_df[ate_error_old_df['method'] != 'mono-gt'] 
                    ate_error_old_df = ate_error_old_df[ate_error_old_df['method'] != 'per-class'] 
                    ate_error_old_df = ate_error_old_df[ate_error_old_df['method'] != 'per-instance'] 
                except:
                    pass   

                try:
                    relative_error_old_df = relative_error_old_df[relative_error_old_df['method'] != 'virtual-gt']
                    relative_error_old_df = relative_error_old_df[relative_error_old_df['method'] != 'mono-gt']
                    relative_error_old_df = relative_error_old_df[relative_error_old_df['method'] != 'global']
                    relative_error_old_df = relative_error_old_df[relative_error_old_df['method'] != 'per-class']
                    relative_error_old_df = relative_error_old_df[relative_error_old_df['method'] != 'per-instance']
                except:
                    pass              
                ate_error_df =  pd.concat([ate_error_old_df, ate_error_df])
                relative_error_df = pd.concat([relative_error_old_df, relative_error_df]) 
            
            # Mean #
            ate_error_df.columns.values[0] = scene_name
            ate_error_df = ate_error_df.drop('scene', axis=1)
        

            relative_error_df.columns.values[0] = scene_name
            relative_error_df = relative_error_df.drop('scene', axis=1)
            
            durations_df.columns.values[0] = scene_name
            durations_df = durations_df.drop('scene', axis=1)
            
            # Format #
            ate_error_df = ate_error_df.round(4)
            relative_error_df = relative_error_df.round(4)
            durations_df = durations_df.round(4)
            
            # Fix combined ate df #
            data_ate_rmse = ate_error_df['abs_trans_rmse'].tolist()
            data_ate_rmse.insert(0, scene_name)
            ate_rmse_all_scenes_list.append(data_ate_rmse)

            data_durations = durations_df['mean'].tolist()
            data_durations.insert(0, scene_name)
            durations_list.append(data_durations)        

            data_durations = durations_df['std'].tolist()
            data_durations.insert(0, scene_name)
            durations_list_std.append(data_durations)      

            data_ate_mean = ate_error_df['abs_trans_mean'].tolist()
            data_ate_mean.insert(0, scene_name)
            ate_mean_all_scenes_list.append(data_ate_mean)

            data_ate_max = ate_error_df['abs_trans_max'].tolist()
            data_ate_max.insert(0, scene_name)
            ate_max_all_scenes_list.append(data_ate_max)

            # Fix combined relative df #
            data_relative_rmse = relative_error_df['trans_rmse'].tolist()
            data_relative_rmse.insert(0, scene_name)
            relative_trans_rmse_all_scenes_list.append(data_relative_rmse)

            data_relative_mean = relative_error_df['trans_mean'].tolist()
            data_relative_mean.insert(0, scene_name)
            relative_trans_mean_all_scenes_list.append(data_relative_mean)

            data_relative_max = relative_error_df['trans_max'].tolist()
            data_relative_max.insert(0, scene_name)
            relative_trans_max_all_scenes_list.append(data_relative_max)

            data_relative_rmse = relative_error_df['rot_rmse'].tolist()
            data_relative_rmse.insert(0, scene_name)
            relative_rot_rmse_all_scenes_list.append(data_relative_rmse)

            data_relative_mean = relative_error_df['rot_mean'].tolist()
            data_relative_mean.insert(0, scene_name)
            relative_rot_mean_all_scenes_list.append(data_relative_mean)

            data_relative_max = relative_error_df['rot_max'].tolist()
            data_relative_max.insert(0, scene_name)
            relative_rot_max_all_scenes_list.append(data_relative_max)
            
            # std #
            ate_error_std_df.columns.values[0] = scene_name
            ate_error_std_df = ate_error_std_df.drop('scene', axis=1)

            relative_error_std_df.columns.values[0] = scene_name
            relative_error_std_df = relative_error_std_df.drop('scene', axis=1)

            # Format #
            ate_error_std_df = ate_error_std_df.round(6)
            relative_error_std_df = relative_error_std_df.round(6)

            # Fix combined ate df #
            data_ate_rmse_std = ate_error_std_df['abs_trans_rmse'].tolist()
            data_ate_rmse_std.insert(0, scene_name)
            ate_rmse_std_all_scenes_list.append(data_ate_rmse_std)

            data_ate_mean_std = ate_error_std_df['abs_trans_mean'].tolist()
            data_ate_mean_std.insert(0, scene_name)
            ate_mean_std_all_scenes_list.append(data_ate_mean_std)

            data_ate_max_std = ate_error_std_df['abs_trans_max'].tolist()
            data_ate_max_std.insert(0, scene_name)
            ate_max_std_all_scenes_list.append(data_ate_max_std)

            # Fix combined relative df #
            data_relative_rmse_std = relative_error_std_df['trans_rmse'].tolist()
            data_relative_rmse_std.insert(0, scene_name)
            relative_trans_rmse_std_all_scenes_list.append(data_relative_rmse_std)

            data_relative_mean_std = relative_error_std_df['trans_mean'].tolist()
            data_relative_mean_std.insert(0, scene_name)
            relative_trans_mean_std_all_scenes_list.append(data_relative_mean_std)

            data_relative_max_std = relative_error_std_df['trans_max'].tolist()
            data_relative_max_std.insert(0, scene_name)
            relative_trans_max_std_all_scenes_list.append(data_relative_max_std)

            data_relative_rmse_std = relative_error_std_df['rot_rmse'].tolist()
            data_relative_rmse_std.insert(0, scene_name)
            relative_rot_rmse_std_all_scenes_list.append(data_relative_rmse_std)

            data_relative_mean_std = relative_error_std_df['rot_mean'].tolist()
            data_relative_mean_std.insert(0, scene_name)
            relative_rot_mean_std_all_scenes_list.append(data_relative_mean_std)

            data_relative_max_std = relative_error_std_df['rot_max'].tolist()
            data_relative_max_std.insert(0, scene_name)
            relative_rot_max_std_all_scenes_list.append(data_relative_max_std)
            
            """
            # Find the row with the lowest value after column 4
            ate_error_df_ = ate_error_df.iloc[:, 3:]
            min_vals_ate = ate_error_df_.idxmin()
            relative_error_df_ = relative_error_df.iloc[:, 3:]
            min_vals_relative = relative_error_df_.idxmin()

            fig, ax = plt.subplots()

            ax.axis('tight')
            ax.axis('off')
            table = ax.table(cellText=ate_error_df.values,
                     colLabels=ate_error_df.columns,
                     loc='center')

            for (row, col), cell in table.get_celld().items():
                if col >= 3 and row - 1 == min_vals_ate[col - 3]:
                    cell.set_text_props(fontproperties=fm.FontProperties(weight='bold'))
            fig.tight_layout(pad=0) 
            #fig.savefig(f'./results/ate_{scene_name}.pdf')
            fig.savefig(f'./results/processed_results/ate_{scene_name}.png', dpi=500, bbox_inches='tight', pad_inches=0)
            plt.close() 

            fig, ax = plt.subplots()

            ax.axis('tight')
            ax.axis('off')
            table = ax.table(cellText=relative_error_df.values,
                     colLabels=relative_error_df.columns,
                     loc='center')

            for (row, col), cell in table.get_celld().items():
                if col >= 3 and row - 1 == min_vals_relative[col - 3]:
                    cell.set_text_props(fontproperties=fm.FontProperties(weight='bold'))

            #fig.savefig(f'./results/relative_error_{scene_name}.pdf')
            fig.tight_layout(pad=0) 
            fig.savefig(f'./results/processed_results/relative_error_{scene_name}.png', dpi=500, bbox_inches='tight', pad_inches=0)
            plt.close()      
            """

#columns = ['scene', 'global', 'virtual', 'virtual-gt', 'mono', 'mono-gt']
#columns = ['scene', 'per-instance', 'per-class', 'global']
columns = ['scene', 'per-instance', 'per-class', 'global', 'virtual', 'mono']
#columns = ['scene', 'global', 'virtual', 'virtual-gt', 'mono', 'mono-gt', 'per-instance', 'per-class']
#columns = ['scene', 'global', 'virtual', 'mono', 'per-instance', 'per-class']
#columns = ['scene', 'per-instance', 'per-class', 'global', 'virtual', 'mono']
#columns = ['scene', 'global', 'virtual', 'mono', 'per-instance', 'per-class']

ate_rmse_all_scenes_df = pd.DataFrame(ate_rmse_all_scenes_list, columns=columns)
ate_mean_all_scenes_df = pd.DataFrame(ate_mean_all_scenes_list, columns=columns)
ate_max_all_scenes_df = pd.DataFrame(ate_max_all_scenes_list, columns=columns)

relative_trans_rmse_all_scenes_df = pd.DataFrame(relative_trans_rmse_all_scenes_list, columns=columns)
relative_trans_mean_all_scenes_df = pd.DataFrame(relative_trans_mean_all_scenes_list, columns=columns)
relative_trans_max_all_scenes_df = pd.DataFrame(relative_trans_max_all_scenes_list, columns=columns)

relative_rot_rmse_all_scenes_df = pd.DataFrame(relative_trans_rmse_all_scenes_list, columns=columns)
relative_rot_mean_all_scenes_df = pd.DataFrame(relative_rot_mean_all_scenes_list, columns=columns)
relative_rot_max_all_scenes_df = pd.DataFrame(relative_rot_max_all_scenes_list, columns=columns)

dfs = [
    ate_rmse_all_scenes_df, 
    ate_mean_all_scenes_df, 
    ate_max_all_scenes_df, 
    relative_trans_rmse_all_scenes_df,
    relative_trans_mean_all_scenes_df, 
    relative_trans_max_all_scenes_df, 
    relative_rot_rmse_all_scenes_df,
    relative_rot_mean_all_scenes_df, 
    relative_rot_max_all_scenes_df,
]

path = './processed_slam/combined/'
if not os.path.exists(path):
    os.makedirs(path)

file_names = [
    'ate_rmse.png', 
    'ate_mean.png', 
    'ate_max.png', 
    'relative_trans_rmse.png', 
    'relative_trans_mean.png',
    'relative_trans_max.png', 
    'relative_rot_rmse.png', 
    'relative_rot_mean.png', 
    'relative_rot_max.png'
]

remove_gt = True

# Find the row with the lowest value after column 4
i = 0
for df, file_name in zip(dfs, file_names):
    try: 
        if remove_gt:
            df.drop('virtual-gt', axis=1, inplace=True)
            #df.drop('virtual', axis=1, inplace=True)
            df.drop('mono-gt', axis=1, inplace=True)
            #df.drop('mono', axis=1, inplace=True)
            pass
    except:
        pass
    
    try:
        df.drop(df[df['scene'] == 'scene0568_02'].index, inplace=True)
        df.drop(df[df['scene'] == 'scene0063_00'].index, inplace=True)
        df.drop(df[df['scene'] == 'scene0647_00'].index, inplace=True)
    except:
        pass

    import time
    time.sleep(0.0001)    

    avg_row = df.iloc[:, 1:].mean()
    
    # Create a new row with 'avg' in the first column and the averages in the subsequent columns
    new_row = ['mean'] + avg_row.tolist()


    df = pd.concat([df, pd.DataFrame([new_row], columns=columns)], ignore_index=True)
    # Append the new row to the DataFrame
    df = df.round(4)
    
    df_ = df.iloc[:, 1:]
    min_vals = df_.idxmin(axis=1)
    min_vals_ids = [df.columns.get_loc(min_val) for min_val in min_vals]  # Per row find the min col
    fig, ax = plt.subplots()
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=df.values, colLabels=df.columns, loc='center')
    for (row, col), cell in table.get_celld().items():
        if row != 0:
            if min_vals_ids[row - 1] == col:
                cell.set_text_props(fontproperties=fm.FontProperties(weight='bold'))


    if i == 0 or i == 3:
        print(df)

    fig.tight_layout(pad=0)
    fig.savefig(path + file_name, dpi=500, bbox_inches='tight', pad_inches=0)
    plt.close()
    i += 1

ate_rmse_std_all_scenes_df = pd.DataFrame(ate_rmse_std_all_scenes_list, columns=columns)
ate_mean_std_all_scenes_df = pd.DataFrame(ate_mean_std_all_scenes_list, columns=columns)
ate_max_std_all_scenes_df = pd.DataFrame(ate_max_std_all_scenes_list, columns=columns)

relative_trans_rmse_std_all_scenes_df = pd.DataFrame(relative_trans_rmse_std_all_scenes_list, columns=columns)
relative_trans_mean_std_all_scenes_df = pd.DataFrame(relative_trans_mean_std_all_scenes_list, columns=columns)
relative_trans_max_std_all_scenes_df = pd.DataFrame(relative_trans_max_std_all_scenes_list, columns=columns)

relative_rot_rmse_std_all_scenes_df = pd.DataFrame(relative_trans_rmse_std_all_scenes_list, columns=columns)
relative_rot_mean_std_all_scenes_df = pd.DataFrame(relative_rot_mean_std_all_scenes_list, columns=columns)
relative_rot_max_std_all_scenes_df = pd.DataFrame(relative_rot_max_std_all_scenes_list, columns=columns)

dfs_std = [
    ate_rmse_std_all_scenes_df, ate_mean_std_all_scenes_df, ate_max_std_all_scenes_df,
    relative_trans_rmse_std_all_scenes_df, relative_trans_mean_std_all_scenes_df, relative_trans_max_std_all_scenes_df,
    relative_rot_rmse_std_all_scenes_df, relative_rot_mean_std_all_scenes_df, relative_rot_max_std_all_scenes_df
]
file_names = [
    'ate_rsme_std.png', 'ate_mean_std.png', 'ate_max_std.png', 'relative_trans_rmse_std.png',
    'relative_trans_mean_std.png', 'relative_trans_max_std.png', 'relative_rot_rmse_std.png',
    'relative_rot_mean_std.png', 'relative_rot_max_std.png'
]

remove_gt = True


# Find the row with the lowest value after column 4
i = 0 
for df, file_name in zip(dfs_std, file_names):
    try:
        df.drop(df[df['scene'] == 'scene0568_02'].index, inplace=True)
        df.drop(df[df['scene'] == 'scene0063_00'].index, inplace=True)
        df.drop(df[df['scene'] == 'scene0647_00'].index, inplace=True)
    except:
        pass

    import time
    time.sleep(0.0001)   
   

    df_ = df.iloc[:, 1:]
    df = df.round(4)
    df = pd.concat([df], ignore_index=True)

    min_vals = df_.idxmin(axis=1)
    min_vals_ids = [df.columns.get_loc(min_val) for min_val in min_vals]  # Per row find the min col

    fig, ax = plt.subplots()

    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=df.values, colLabels=df.columns, loc='center')

    for (row, col), cell in table.get_celld().items():
        if row != 0:
            if min_vals_ids[row - 1] == col:
                cell.set_text_props(fontproperties=fm.FontProperties(weight='bold'))

    if i == 0 or i == 3:
        print(df)
    
    i += 1
    fig.tight_layout(pad=0)
    fig.savefig(path + file_name, dpi=500, bbox_inches='tight', pad_inches=0)
    plt.close()

durations_all_df = pd.DataFrame(durations_list, columns=columns)
durations_all_std_df = pd.DataFrame(durations_list_std, columns=columns)

dfs = [
    durations_all_df, durations_all_std_df
]
file_names = [
    'durations_mean.png',
    'durations_std.png',
]

# Find the row with the lowest value after column 4
for df, file_name in zip(dfs, file_names):
    try:
        df.drop(df[df['scene'] == 'scene0568_02'].index, inplace=True)
        df.drop(df[df['scene'] == 'scene0063_00'].index, inplace=True)
        df.drop(df[df['scene'] == 'scene0647_00'].index, inplace=True)
    except:
        pass

    import time
    time.sleep(0.0001)  

    df_ = df.iloc[:, 1:]
    df = df.round(4)
    df = pd.concat([df], ignore_index=True)
    min_vals = df_.idxmin(axis=1)
    min_vals_ids = [df.columns.get_loc(min_val) for min_val in min_vals]  # Per row find the min col

    fig, ax = plt.subplots()

    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=df.values, colLabels=df.columns, loc='center')

    for (row, col), cell in table.get_celld().items():
        if row != 0:
            if min_vals_ids[row - 1] == col:
                cell.set_text_props(fontproperties=fm.FontProperties(weight='bold'))

    print(df)

    print(df[['per-instance', 'per-class', 'global', 'virtual', 'mono']].mean())
    fig.tight_layout(pad=0)
    fig.savefig(path + file_name, dpi=500, bbox_inches='tight', pad_inches=0)
    plt.close()

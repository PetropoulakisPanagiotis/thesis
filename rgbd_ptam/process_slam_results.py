import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.font_manager as fm
import os
import numpy as np

parent_dir = './results'

ate_errors = []
relative_errors = []

ate_rmse_all_scenes_list = []
ate_mean_all_scenes_list = []
ate_max_all_scenes_list = []

relative_trans_rmse_all_scenes_list = []
relative_trans_mean_all_scenes_list = []
relative_trans_max_all_scenes_list = []

relative_rot_rmse_all_scenes_list = []
relative_rot_mean_all_scenes_list = []
relative_rot_max_all_scenes_list = []

for folder in os.listdir(parent_dir):
    if os.path.isdir(os.path.join(parent_dir, folder)):
        if 'scene' in folder:
            scene_name = folder
            
            ate_error_df = pd.read_csv(os.path.join(parent_dir, folder, 'ate_error.csv'))
            relative_error_df = pd.read_csv(os.path.join(parent_dir, folder, 'relative_error.csv'))
            if 'abs_trans_rmse' not in ate_error_df.columns:
                continue
            
            ate_error_df.columns.values[0] = scene_name
            ate_error_df = ate_error_df.drop('scene', axis=1)
           
            relative_error_df.columns.values[0] = scene_name
            relative_error_df = relative_error_df.drop('scene', axis=1)

            # Format #
            ate_error_df = ate_error_df.round(4)
            relative_error_df = relative_error_df.round(4)

            # Fix combined ate df #
            data_ate_rmse = ate_error_df['abs_trans_rmse'].tolist()
            data_ate_rmse.insert(0, scene_name)
            ate_rmse_all_scenes_list.append(data_ate_rmse)

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


columns=['scene', 'global', 'virtual', 'virtual-gt', 'mono', 'mono-gt']

ate_rmse_all_scenes_df = pd.DataFrame(ate_rmse_all_scenes_list, columns=columns)
ate_mean_all_scenes_df = pd.DataFrame(ate_mean_all_scenes_list, columns=columns)
ate_max_all_scenes_df = pd.DataFrame(ate_max_all_scenes_list, columns=columns)

relative_trans_rmse_all_scenes_df = pd.DataFrame(relative_trans_rmse_all_scenes_list, columns=columns)
relative_trans_mean_all_scenes_df = pd.DataFrame(relative_trans_mean_all_scenes_list, columns=columns)
relative_trans_max_all_scenes_df = pd.DataFrame(relative_trans_max_all_scenes_list, columns=columns)

relative_rot_rmse_all_scenes_df = pd.DataFrame(relative_trans_rmse_all_scenes_list, columns=columns)
relative_rot_mean_all_scenes_df = pd.DataFrame(relative_rot_mean_all_scenes_list, columns=columns)
relative_rot_max_all_scenes_df = pd.DataFrame(relative_rot_max_all_scenes_list, columns=columns)

dfs = [ate_rmse_all_scenes_df, ate_mean_all_scenes_df, ate_max_all_scenes_df,
       relative_trans_rmse_all_scenes_df, relative_trans_mean_all_scenes_df, relative_trans_max_all_scenes_df,
       relative_rot_rmse_all_scenes_df, relative_rot_mean_all_scenes_df, relative_rot_max_all_scenes_df]

path = './results/processed_results/combined/'
file_names = [
'ate_rsme.png',
'ate_mean.png',
'ate_max.png',
'relative_trans_rsme.png',
'relative_trans_mean.png',
'relative_trans_max.png',
'relative_rot_rsme.png',
'relative_rot_mean.png',
'relative_rot_max.png'
]

# Find the row with the lowest value after column 4
for df, file_name in zip(dfs, file_names):
    df_ = df.iloc[:, 1:]
    min_vals = df_.idxmin(axis=1)
    min_vals_ids = [df.columns.get_loc(min_val) for min_val in min_vals] # Per row find the min col 

    fig, ax = plt.subplots()

    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=df.values,
             colLabels=df.columns,
             loc='center')

    for (row, col), cell in table.get_celld().items():
        if row != 0:
            if min_vals_ids[row - 1] == col:        
                cell.set_text_props(fontproperties=fm.FontProperties(weight='bold'))
    
    fig.tight_layout(pad=0) 
    fig.savefig(path + file_name, dpi=500, bbox_inches='tight', pad_inches=0)
    plt.close()   

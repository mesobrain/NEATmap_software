# 3D-HSFormer parameter
import os

data_name = '20210824_GLX_MHW_C6_16_1'
name = data_name.split('_')
if len(name) == 6:
    name_seg = name[1] + str('_') + name[2] + str('_') + name[3] + str('_') + name[4] + str('_') + name[5]
else:
    print('Recheck the name of the data')
Raw_data_root = r'U:\VISoRData\MHW\MHW-SD-B2\MHW-SD-B2-part2/' + data_name + '/Reconstruction/BrainImage/4.0'
Data_root = r'R:\WeijieZheng\Data_proc_' + data_name
Save_root = r'R:\WeijieZheng\Model_'+ name_seg +'_seg_pre'

Channels = {}
Channels['autofluo'] = '488nm'
Channels['staining'] = '561nm'
Channels['561nm_index'] = 'C3'
Channels['488nm_index'] = 'C2'

Brain = {}
Brain['z_num'] = 75
Brain['height'] = 2500
Brain['weight'] = 3500
Brain['voxel_size'] = 4

Preprocessing = {}
Preprocessing['z_num'] = 64
Preprocessing['cut_size'] = 256
Preprocessing['cut_index_x'] = 470
Preprocessing['cut_index_y'] = 245
Preprocessing['patch_weight_num'] = 10
Preprocessing['patch_height_num'] = 7

Spot_seg_config = {}
Spot_seg_config['scaling_param'] = [1, 40]
Spot_seg_config['gaussian_smoothing_sigma'] = 1
Spot_seg_config['wrapper_param'] = [[1, 0.01], [1, 0.006]] # [[mu1, sigma1], [mu2, sigma2], ...]
Spot_seg_config['save_seg_path'] = ''

Train_config = {}
Train_config['train_total_num'] = 40
Train_config['train_image_path'] = None
Train_config['train_label_path'] = None
Train_config['train_data_path'] = ''

Test_config = {}
Test_config['test_id'] = 30
Test_config['test_path'] = os.path.join('', 'brain_label_64')
Test_config['test_image_path'] = ''
Test_config['test_label_path'] = ''
Test_config['test_save_path'] = ''

Network = {}
Network['train_data'] = ''
Network['test_slice'] = ''
Network['train_valid_ratio'] = 0.9
Network['test_whole_brain'] = ''
Network['save_data_list'] = ''
Network['lr'] = 0.01
Network['max_epochs'] = 10
Network['num_classes'] = 2
Network['save_checkpoint'] = ''
Network['train_batch_size'] = 2
Network['valid_batch_size'] = 1
Network['test_batch_size'] = 1
Network['train_shuffle'] = True
Network['valid_shuffle'] = False
Network['train_num_workers'] = 4
Network['valid_num_workers'] = 1
Network['pin_memory'] = True
Network['loss_ratio'] = 0.5
Network['pre_train'] = False
Network['Pre_train_path'] = ''
Network['save_patch_metrics'] = ''
Network['save_metrics'] = ''
Network['z_spacing'] = 1
Infer = {}
Infer['snapshot_path'] = '/home/weijie/NEATmap/Network/checkpoint/swin_T_checkpoint_22_01_09'
Infer['whole_brain_path'] = '/data/weijie/Test_NEATmap/save'
Infer['whole_brain_list'] = 'pages/Whole_brain_seg/whole_brain_lists'
Infer['swin_in_channels'] = 16

Optimizer = {}
Optimizer['momentum'] = 0.9
Optimizer['SGD_weight_decay'] = 0.0001
Optimizer['Adam_weight_decay'] = 0.005
Optimizer['AdamW_weight_decay'] = 5E-2

Splicing = {}
Splicing['whole_predications_path'] = r'R:\WeijieZheng\Data_prco_20210824_GLX_MHW_C5_13_1'
Splicing['checkpoint_name'] = 'swin_T_checkpoint_22_01_09'
Splicing['save_root'] = Save_root
Splicing['save_splicing_path'] = Save_root

Post = {}
Post['intensity_lower_differ'] = -95
Post['intensity_upper_differ'] = -47
Post['point_min_size'] = 1
Post['big_object_size'] = 3000

Registration = {}
Registration['parameters_root'] = 'R:/WeijieZheng/NEATmap_Code/NEATmap/Registration/parameters'
Registration['Raw_brain_path'] = r'U:\VISoRData\MHW\MHW-SD-B2\MHW-SD-B2-part2\20210830_GLX_MHW_C9_25_1'
Registration['output_path'] = r'R:\WeijieZheng\Model_GLX_MHW_C9_25_1_seg_pre'
Registration['output_name'] = 'brain_registration'
Registration['registration_param'] = 'tp_brain_registration_bspline.txt'
Registration['inverse_param'] = 'tp_inverse_bspline.txt'

Spot_map = {}
Spot_map['group_num'] = 6.25
Spot_map['filter_area_lower'] = 1
Spot_map['filter_area_upper'] = 27

freesia_export_path = r'R:\WeijieZheng\Model_'+ name_seg +'_seg_pre\whole_brain_cell_counts'

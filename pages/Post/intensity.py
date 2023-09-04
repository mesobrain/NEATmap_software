from Environment_ui import *
from scipy import ndimage as ndi

def get_2d_local_voxels(block, yc, xc, rad ):
    """
    """
    xst = xc-rad if xc-rad>=0 else 0
    yst = yc-rad if yc-rad>=0 else 0
    xen = xc+rad+1 if xc+rad+1<=block.shape[1] else block.shape[1]
    yen = yc+rad+1 if yc+rad+1<=block.shape[0] else block.shape[0]

    return block[ yst:yen, xst:xen ]

def get_2d_intensity(blk, label, obj_id, arr_idx,
                    mode='max', max_rad=1, min_rad=4, min_percentile=5):
    """
    """
    block = blk.copy()
    c0, c1 = arr_idx

    # compute bagkground level
    local = get_2d_local_voxels( block, c0, c1, min_rad )
    bg = np.percentile(local, min_percentile )

    if mode=='max':
        local = get_2d_local_voxels( block, c0, c1, max_rad )
        sig = local.max()
    elif mode=='local_mean':
        local = get_2d_local_voxels( block, c0, c1, max_rad )
        sig = local.mean()
    elif mode=='obj_mean':
        sig = ndi.mean( block, label, obj_id )

    #compute delta
    delta = sig - bg
    if delta < 0:
        delta = 0

    return delta, bg

def plot_normalized_counts(intensity_root, save_path, upper_value=800):
    os.makedirs(save_path, exist_ok=True)
    intensity_values = []
    for i in range(1, len(os.listdir(intensity_root)) + 1):
        csv_path = os.path.join(intensity_root, 'intensity_{}.csv'.format(i))
        file = open(csv_path)
        df = pd.read_csv(file)
        intensity = df['intensity'].values
        intensity_values.extend(intensity)
    j = 0    
    while j < len(intensity_values):
        if intensity_values[j] > upper_value:
            del intensity_values[j]
        else:
            j += 1

    intensity_range = (min(intensity_values), max(intensity_values))
    hist, bin_edges = np.histogram(intensity_values, bins=20,
                                    range=intensity_range,
                                    density=False)
    bins = ( bin_edges[:-1] + bin_edges[1:] ) / 2
    savename = os.path.join(save_path, 'whole_brain_detected_neuro_intensity.csv')
    values = pd.DataFrame(columns=['intensity', 'normalized_count'])
    values['intensity'] = bins
    values['normalized_count'] = hist
    values.to_csv(savename, index=False)
    print('Export csv file')

    print('Drawing histigram ...')
    plt.bar(bins, hist, width=10, align='center', alpha=0.7, color='b')

    plt.xlabel('Intensity')
    plt.ylabel('Counts')
    plt.title('Whole brain detected neuro intensity')
    plt.savefig(os.path.join(save_path, 'Intensity.png'))
    return print('Finished')
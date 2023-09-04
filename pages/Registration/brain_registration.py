from Environment_ui import *
from math import ceil

ROOT_DIR = os.path.dirname(__file__)
PARAMETER_DIR = 'pages/Registration/parameters'

def generate_freesia_input(input_path: str, shape, channel, pixel_size, group_size, **kwargs):
    doc = {
        "group_size": group_size,
        "image_path": os.path.split(input_path)[1],
        "images": [],
        "pixel_size": pixel_size,
        "slide_thickness": pixel_size,
        "version": "1.1.2"
    }
    h, w = shape[1], shape[0]

    files = {f for f in os.listdir(input_path) if re.match('Z\d+_C\d+.tif', f) is not None}

    ct = 0
    while ct < 100000 and len(files) > 0:
        name = 'Z{:05d}_C{}.tif'.format(ct, channel)
        if name not in files:
            ct += 1
            continue
        files.remove(name)
        d = {
            "index": ct,
            "height": h,
            "width": w,
            "file_name": name
        }
        doc['images'].append(d)
        ct += 1

    output_file = os.path.join(os.path.split(input_path)[0],
                               'freesia_{}_C{}.json'.format(os.path.split(input_path)[1], channel))
    #with open(output_file, 'w') as fp:
    #    json.dump(doc, fp, indent=2)
    return json.dumps(doc)


def write_freesia2_image(image: sitk.Image, path: str, name: str, pixel_size, group_size):
    doc = {
        "group_size": group_size,
        "image_path": name,
        "images": [],
        "voxel_size": pixel_size,
        #"slide_thickness": pixel_size
    }
    h, w = image.GetSize()[1], image.GetSize()[0]

    image_path = os.path.join(path, name)
    if not os.path.exists(image_path):
        os.mkdir(image_path)
    files = []
    for z in range(image.GetSize()[2]):
        file = '{:04d}_{}.tif'.format(z, pixel_size)
        files.append(os.path.join(image_path, file))
        d = {
            "index": z + 1,
            "height": h,
            "width": w,
            "file_name": file
        }
        doc['images'].append(d)
    sitk.WriteImage(image, files)

    doc['freesia_project'] = {}
    freesia_project = doc['freesia_project']
    freesia_project['transform_2d'] = []
    for i in range(int(ceil(image.GetSize()[2] / group_size))):
        freesia_project['transform_2d'].append({
            "group_index": i,
            "rotation": "0",
            "scale": "1 1",
            "translation": "0 0"})
    freesia_project["transform_3d"] = {
        "rotation": "0 0",
        "scale": "1 1 1",
        "translation": "0 0 0"}
    freesia_project['warp_markers'] = []

    output_file = os.path.join(path, '{}.json'.format(name))
    with open(output_file, 'w') as fp:
        json.dump(doc, fp, indent=4)

def read_freesia2_image(file: str):
    with open(file) as f:
        doc = json.load(f)
    files = []
    for i in doc['images']:
        files.append(os.path.join(os.path.dirname(file), doc['image_path'], i['file_name']))
    return sitk.ReadImage(files)


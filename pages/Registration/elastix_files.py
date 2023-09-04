from Environment_ui import *

PARAMETER_DIR = 'pages/Registration/parameters'
def get_sitk_transform(elastix_param):
    tf_type = elastix_param['Transform'][0]
    image_spec = {'Size': [int(i) for i in elastix_param['Size']],
                  'Origin': [float(i) for i in elastix_param['Origin']],
                  'Spacing': [float(i) for i in elastix_param['Spacing']],
                  'Direction': [float(i) for i in elastix_param['Direction']]}
    ndim = int(elastix_param['FixedImageDimension'][0])

    if tf_type == 'EulerTransform':
        param = [float(i) for i in elastix_param['TransformParameters']]
        center = [float(i) for i in elastix_param['CenterOfRotationPoint']]
        if ndim == 2:
            tf = sitk.Euler2DTransform()
        elif ndim == 3:
            tf = sitk.Euler3DTransform()
        else:
            raise NotImplementedError('cannot resolve transformation {},{}'.format(tf_type, ndim))
        tf.SetCenter(center)
        tf.SetParameters(param)

    elif tf_type == 'AffineTransform':
        param = [float(i) for i in elastix_param['TransformParameters']]
        center = [float(i) for i in elastix_param['CenterOfRotationPoint']]
        tf = sitk.AffineTransform(ndim)
        tf.SetCenter(center)
        tf.SetParameters(param)

    elif tf_type == 'BSplineTransform':
        order = int(elastix_param['BSplineTransformSplineOrder'][0])
        param = [float(i) for i in elastix_param['TransformParameters']]
        tf = sitk.BSplineTransform(ndim, order)
        mesh_size =[int(i) - 3 for i in elastix_param['GridSize']]
        physical_dim = [float(elastix_param['GridSpacing'][i]) * mesh_size[i] for i in range(ndim)]
        origin = [float(elastix_param['GridOrigin'][i]) + float(elastix_param['GridSpacing'][i]) for i in range(ndim)]
        direction = [float(i) for i in elastix_param['GridDirection']]
        tf.SetTransformDomainMeshSize(mesh_size)
        tf.SetTransformDomainOrigin(origin)
        tf.SetTransformDomainPhysicalDimensions(physical_dim)
        tf.SetTransformDomainDirection(direction)
        tf.SetParameters(param)

    else:
        raise NotImplementedError('cannot resolve transformation {},{}'.format(tf_type, ndim))

    return tf, image_spec

def get_align_transform(fixed, moving, parameter_files, visormap_param, fixed_mask=None, moving_mask=None,
                        fixed_points=None, moving_points=None, rigidity_mask=None, inverse_transform=False,
                        initial_transform=None, multichannel=False):
    with tempfile.TemporaryDirectory() as ELASTIX_TEMP:
        elastix = sitk.ElastixImageFilter()
        elastix.SetOutputDirectory(ELASTIX_TEMP)
        params = sitk.VectorOfParameterMap()
        for p in parameter_files:
            param = sitk.ReadParameterFile(p)
            size = 1
            for s in moving.GetSize():
                size *= s
            if len(moving.GetSize()) == 2:
                param['NumberOfSpatialSamples'] = [str(int(max(moving.GetSize()[0] * moving.GetSize()[1] / 2048 * pow(4, i), 2048))) for i in range(4)]
            if rigidity_mask is not None:
                mask_path = os.path.join(ELASTIX_TEMP, 'rigidity_mask.mha')
                sitk.WriteImage(rigidity_mask, mask_path)
                param['MovingRigidityImageName'] = [mask_path]
            if multichannel:
                if param['Registration'][0] == 'MultiMetricMultiResolutionRegistration':
                    m = [*param['Metric']]
                    for i in range(1, fixed.GetSize()[2]):
                        m = [param['Metric'][0], *m]
                    param['Metric'] = m
                m = {'FixedImagePyramid': [], 'MovingImagePyramid': [], 'Interpolator': [], 'ImageSampler': []}
                for i in range(0, len(param['Metric'])):
                    for k in m:
                        m[k] = [*m[k], param[k][0]]
                for k in m:
                    param[k] = m[k]
            params.append(param)

        elastix.SetParameterMap(params)
        if multichannel:
            for c in range(fixed.GetSize()[2]):
                elastix.AddFixedImage(fixed[:, :, c])
                elastix.AddMovingImage(moving[:, :, c])
            for i in range(len(param['Metric']) - fixed.GetSize()[2]):
                elastix.AddFixedImage(fixed[:, :, 0])
                elastix.AddMovingImage(moving[:, :, 0])
        else:
            elastix.SetFixedImage(fixed)
            elastix.SetMovingImage(moving)
        if fixed_mask is not None:
            elastix.SetFixedMask(fixed_mask)
        if moving_mask is not None:
            elastix.SetMovingMask(moving_mask)
        if fixed_points is not None and moving_points is not None:
            elastix.SetFixedPointSetFileName(fixed_points)
            elastix.SetMovingPointSetFileName(moving_points)
        if initial_transform is not None:
            elastix.SetInitialTransformParameterFileName(initial_transform)
        s = elastix.Execute()
        tf_par = elastix.GetTransformParameterMap()
        #transformix = sitk.TransformixImageFilter()
        #transformix.SetOutputDirectory(ELASTIX_TEMP.name)
        #transformix.ComputeDeformationFieldOn()
        #transformix.SetTransformParameterMap(tf_par)
        #transformix.Execute()
        transform = None
        transforms = []
        for p in tf_par:
            tf, im = get_sitk_transform(p)
            if transform is None:
                transform = sitk.Transform(tf.GetDimension(), sitk.sitkComposite)
            transforms.append(tf)
        transforms.reverse()
        if len(transforms) == 1:
            transform = transforms[0]
        else:
            for t in transforms:
                transform.AddTransform(t)
        #df = sitk.ReadImage(os.path.join(ELASTIX_TEMP.name, 'deformationField.mhd'))
        #df = sitk.Compose(sitk.VectorIndexSelectionCast(df, 0),
        #                  sitk.VectorIndexSelectionCast(df, 1))
        #df = sitk.Cast(df, sitk.sitkVectorFloat64)
        if inverse_transform:
            ct = 0
            for p in tf_par:
                file = os.path.join(ELASTIX_TEMP, 'TransformParameters.{}.txt'.format(ct))
                if ct > 0:
                    p['InitialTransformParametersFileName'] = [os.path.join(ELASTIX_TEMP, 'TransformParameters.{}.txt'.format(ct - 1)).replace('\\', '/')]
                #p['Size'] = [str(k) for k in moving.GetSize()]
                elastix.WriteParameterFile(p, file)
                ct += 1
            out, inv = get_align_transform(moving, fixed, [os.path.join(PARAMETER_DIR, visormap_param['inverse_param'])],
                                           initial_transform=file)
            return s, transform, inv
    return s, transform
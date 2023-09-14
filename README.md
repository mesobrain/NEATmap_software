<div align="center">
  
# NEATmap documentation

</div>

_**NEATmap: a high-efficiency deep learning approach for whole mouse brain neuronal activity trace mapping**_

Weijie Zheng*, Huawei Mu*, Zhiyi Chen, Jiajun Liu, Debin Xia, Yuxiao, Cheng, Qi Jing, Jin Tang, Guo-Qiang Bi, Feng Wu and Hao Wang. NEATmap: a high-efficiency deep learning approach for whole mouse brain neuronal activity trace mapping submitted to *National Science Review*.

\* equal contribution

<p align="center">
<img src="doc\Html\NEATmap_demo.PNG"
height="300">
</p>

The explanation is as follows:

## Content

* [**1. Abstract & Availability**](#abstract)
* [**2. Installation & Running**](#installation)
* [**3. Settings**](#settings)
* [**4. Data preprocessing**](#data_preprocessing)
* [**5. Whole brain segmentation**](#whole_brain_segmentation)
* [**6. Splice & Post processing**](#Splice)
* [**7. Registration & Analysis**](#registration)
<br />

<div style="text-align: justify">

### 1.	Abstract & Availability<a name="abstract"></a> 
Quantitative analysis of activated neurons in mice brains by a specific stimulation has long been a primary step to locate the distribution of responsive neurons across the brain. However, it’s challenging to comprehensively and consistently analyze the neuronal activation in whole brains of large cohorts of mice from many Terabytes of volumetric imaging datasets. Here, we introduce **NEATmap**, a deep-learning based high-efficiency, high-precision, and user-friendly software for whole brain **NE**uronal **A**ctivity **T**race **map**ping by automated segmentation and quantitative analysis of immunofluorescence labeled c-Fos+ neurons. We applied NEATmap to study the brain-wide differentiated neuronal activation in response to physical and psychological stressors in cohorts of mice.
 <br />
<br />
Availability and implementation:
  
The source code for all modules of NEATmap is implemented in Python.   

Code source, tutorial, documentation are available at: https://github.com/mesobrain/NEATmap_code.

Whole-brain test dataset is available at: https://zenodo.org/record/8133486.
<br />

### 2. Installation $ Running <a name="installation"></a> 

The list of names of python libraries required by NEATmap can be found in the environment file `NEATmap_software.yaml`.
    
    $ conda env create -f NEATmap_software.yaml

Similarly, you can install other python dependencies via pip and use

    $ pip install name

Running:

After completing the above installation, execute the command to enter NEATmap.

    $ python NEATmap_ui.py
<br />

### 3. Settings <a name="settings"></a> 

Click the **Settings** button to display the parameter setting interface. If the user needs to modify the parameters, make the changes and click the **Save** button.

<p align="center">
<img src="doc\Html\settings_demo.png"
height=300>
<img src="doc\Html\settings.png"
height=300>
</p>
<br />

### 4. Data preprocessing <a name="data_preprocessing"></a> 

Clicking the **Data preprocessing** button enables the generation of both volumetric images and patch images (sub-volumes).

<p align="center">
<img src="doc\Html\datapreprocessing_demo.png"
height=250>
<img src="doc\Html\datapreprocessing.png"
height=250>
</p>
<br />

#### Diagram representation:

<p align="center">
<img src="doc\Html\datapreprocessing_result.png"
height=300>
</p>

### 5. Whole brain segmentation <a name="whole_brain_segmentation"></a> 

Clicking on **Whole brain segmentation** enables automatic segmentation of immunolabeled signal (c-Fos) across the whole brain.

<p align="center">
<img src="doc\Html\wholebrainseg_demo.png"
height=250>
<img src="doc\Html\wholebrainseg.png"
height=250>
</p>
<br />

#### 3D-HSFormer architecture:

<p align="center">
<img src="doc\Html\3D-HSFormer.png"
height=300>
</p>

### 6. Splice & Post processing <a name="Splice"></a> 

Clicking on **Splice** followed by **Post processing** results in the segmented map of immunolabeled signal (c-Fos) across the whole brain.

<p align="center">
<img src="doc\Html\Splice&post_demo.png"
height=300>
</p>

#### Diagram representation:

<p align="center">
<img src="doc\Html\Splice&post.png"
height=300>
</p>

### 7. Registration & Analysis <a name="registration"></a> 

Clicking on **Registration** followed by **Analysis** enables the registration of the whole brain to the [Allen Common Coordinate Framework atlas](https://atlas.brain-map.org/) and provides information about the spatial coordinates, number, morphology, and other features of cells across the hierarchical regions of the brain.

<p align="center">
<img src="doc\Html\registration&analysis_demo.png"
height=300>
</p>

#### Results:

<p align="center">
<img src="doc\Html\registration.png"
height=250>
<img src="doc\Html\analysis.png"
height=250>
</p>
<br />

## Supplementary instruction

Detailed instructions for specific operations can be found in the **User_guide.pdf** file located in the **doc** directory. The usage tutorial for the NEATmap software can be found in the **Tutorial_video** folder under the file name **neatmap_tutorial.mp4**. Additionally, the **seg-display.mp4** file demonstrates the segmentation and cell counting results of NEATmap.

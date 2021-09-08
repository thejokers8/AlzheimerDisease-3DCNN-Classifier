# AlzheimerDisease-3DCNN-Classifier
## What is it?
Here is the documentation about the construction and methods used in medical image classification with MRI and 3D convolutional nueral networks. It's important to you'd been prior expirience with neuroscience software as ANTs, SPM12, FREESURFER, FSL or even Nipype among others. As a general view the project has 3 stages: Preprocessing pipeline from Clinica `<T1-linnear>`, just for anatomical images and linnear processing, model construction and trainning with `<deep learning prepare data>` from Clinica too and interpretability analysis. The whole project was coded in python currently we just work with T1-Weighted MRI scans by the ADNI database

# *third-partly software*
First off all we need to set up our environment in such way we run different software plataforms to construct our pipelines and to assess our partial results. So this software resources it's going to be a strong and powerfull help during the nexts developing tasks the table bellow sumarizes the main packages used in neuroscience apllications:

Package name  | Documentation
------------  | -------------
Fsl           | https://fsl.fmrib.ox.ac.uk/fsl/fslwiki - automatic!

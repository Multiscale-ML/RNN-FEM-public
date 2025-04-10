The main function of the Preprocessing script is to generate high-fidelity RVE models with applied strain series and write corresponding ABAQUS input files. Specifically, the preprocessing code generates the geometry of RVE models, assigns material properties, creates a mesh, and generates periodic boundary conditions (PBC) of RVEs with applied macroscopic strain series. 

It should be pointed out that both the Preprocessing and Postprocessing-Homogenization code need to run in the environment of ABAQUS CAE. To run these scripts in ABAQUS CAE, click on "File" and "Run Script" in sequence. Finally, select the script that needs to be run and click "OK" to run the script. Our paper provides a detailed explanation of the preprocessing code, including the technical details for generating periodic boundary conditions.

The figure below shows an example of a RVE model generated by our preprocessing algorithm:
<img src="https://github.com/Multiscale-ML/RNN-FEM/assets/139171109/9cceed6f-4f9f-4890-a9ee-c8289c1936b3" width="550" height="400">


The main goal of the Postprocessing-Homogenization code is to extract stress data at the integration points from the ABAQUS output (.odb) files after the high-fidelity simulations are completed. Subsequently, the script calculates the homogenized response of each RVE.

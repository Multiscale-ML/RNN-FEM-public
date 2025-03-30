**1. Documentation of UMAT of GRU**

UMAT (User Materials), is a user-defined material interface provided by the commercial finite element software ABAQUS. UMAT is used to implement non-built-in constitutive models in ABAQUS which uses the programming language FORTRAN. We implement the trained GRU model in UMAT, which will enable ABAQUS to directly use our GRU model as a surrogate model for composite materials during simulations.

Our UMAT file is divided into three parts: the first part is the matrix operation subroutine, the second part is the trained GRU model, and the third part is the UMAT subroutine. We define the necessary subroutines in the first part, such as matrix addition and subtraction, MinMax Scaler, Min Max Scaler Inverse, and activation function subroutines. In the second part, we call the subroutines of the first part, declare all variables, and build each GRU layer step by step. In the third part, we call the GRU subroutine to complete UMAT, mainly to input strain and hidden state from the previous step into the GRU subroutine and return  stress.

Below is an example simulation implemented by our UMAT of trained GRU:
![image](https://github.com/Multiscale-ML/RNN-FEM/assets/139171109/98ba0bf6-5ccb-48fa-bc2e-c909cc7f2a58)



**2. Documentation of Data Writer**

The function of the data writer is to directly write the model parameters of the trained GRU into the UMAT file. Due to the complex structure of GRU, we have a large number of model parameters, thus writing the parameters directly into the same file makes it more convenient to use. In the model data loading section of our code, we have pre-written the names of each layer's parameters as keywords. Our data writer will read every line of code and write the corresponding model parameters to their respective positions when the model parameter name is detected.


**3. Documentation of UVARM of von Mises Strain**

UVARM (User VARables) is a subroutine provided by ABAQUS that is used to track non-built-in variables in simulations and generate their contour plots in post-processing. Our UVARM subroutine will use the GETVRM subroutine to retrieve the strain vector in each step of the simulation, compute the von Mises strain in the current time step, and return this variable. We can merge UVARM with UMAT into one FORTRAN file to use.

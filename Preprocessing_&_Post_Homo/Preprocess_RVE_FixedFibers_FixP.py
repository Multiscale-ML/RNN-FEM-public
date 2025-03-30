# Preprocessing script for generating RVE with Fixed fibers and Fixed parameters
# Function: Preprocessing (Running in ABAQUS CAE)
# Developed by Yijing Zhou
# 12/05/2023 re-write section 7 12/11/2023


import numpy as np
import random

from abaqus import *
from abaqusConstants import *


## 0. Parameters and load datasets

num_rve = 1
dim_rve = 1.0
num_fiber = 1
num_time_steps = 50
size_mesh = 0.075
ini_Inc = 0.1

strain_hat_dataset = np.load("strain_hat_dataset.npy")


## 1. Generate models and parts

for i in range(num_rve):
    # Generate models
    mdb.Model(name='Model-'+str(i+1), modelType=STANDARD_EXPLICIT)
    # Generate parts
    s = mdb.models['Model-'+str(i+1)].ConstrainedSketch(name='__profile__', sheetSize=dim_rve)
    g, v, d, c = s.geometry, s.vertices, s.dimensions, s.constraints
    s.setPrimaryObject(option=STANDALONE)
    s.rectangle(point1=(dim_rve / 2, dim_rve / 2), point2=(-dim_rve / 2, -dim_rve / 2))
    p = mdb.models['Model-'+str(i+1)].Part(name='Part-1', dimensionality=THREE_D, 
        type=DEFORMABLE_BODY)
    p = mdb.models['Model-'+str(i+1)].parts['Part-1']
    p.BaseSolidExtrude(sketch=s, depth=dim_rve)
    s.unsetPrimaryObject()
    del mdb.models['Model-'+str(i+1)].sketches['__profile__']


## 2. Generate fibers with different positions

# Generate
for i in range(num_rve):
    p = mdb.models['Model-'+str(i+1)].parts['Part-1']
    f, e = p.faces, p.edges
    t = p.MakeSketchTransform(sketchPlane=f[4], sketchUpEdge=e[0], 
        sketchPlaneSide=SIDE1, sketchOrientation=RIGHT, origin=(0.0, 0.0, 0.1))
    s1 = mdb.models['Model-'+str(i+1)].ConstrainedSketch(name='__profile__', sheetSize=3.0, 
        gridSpacing=0.07, transform=t)

    # Draw geometry on a cross-section
    for j in range(num_fiber):
        g, v, d, c = s1.geometry, s1.vertices, s1.dimensions, s1.constraints
        s1.setPrimaryObject(option=SUPERIMPOSE)
        p.projectReferencesOntoSketch(sketch=s1, filter=COPLANAR_EDGES)
        s1.CircleByCenterPerimeter(center=(0, 0),\
                                   point1=(0, 0.35))
        f1, e1 = p.faces, p.edges

    # Solid extrude
    p.SolidExtrude(sketchPlane=f1[4], sketchUpEdge=e1[0], sketchPlaneSide=SIDE1, 
        sketchOrientation=RIGHT, sketch=s1, depth=dim_rve, flipExtrudeDirection=ON, 
        keepInternalBoundaries=ON)
    s1.unsetPrimaryObject()
    del mdb.models['Model-'+str(i+1)].sketches['__profile__']

session.viewports['Viewport: 1'].view.setProjection(projection=PERSPECTIVE)


## 3. Input materials / Create sections

# Input materials of fiber and matrix
for i in range(num_rve):
    # Materials
    mdb.models['Model-'+str(i+1)].Material(name='Material-fiber')
    mdb.models['Model-'+str(i+1)].materials['Material-fiber'].Elastic(table=((324000, 0.1), ))
    mdb.models['Model-'+str(i+1)].Material(name='Material-matrix')
    mdb.models['Model-'+str(i+1)].materials['Material-matrix'].Elastic(table=((71700, 0.33), ))
    mdb.models['Model-'+str(i+1)].materials['Material-matrix'].Plastic(scaleStress=None, table=(
    (455, 0.0), (655, 0.208)))
    # Sections
    mdb.models['Model-'+str(i+1)].HomogeneousSolidSection(name='Section-1', 
    material='Material-fiber', thickness=None)
    mdb.models['Model-'+str(i+1)].HomogeneousSolidSection(name='Section-2', 
    material='Material-matrix', thickness=None)


## 4. Assign sections

# Assign section of fibers
for i in range(num_rve):
    for j in range(num_fiber):
        x_j = 0
        y_j = 0
        p = mdb.models['Model-'+str(i+1)].parts['Part-1']
        c = p.cells
        cells = c.findAt(((x_j, y_j, 0),))
        region = p.Set(cells=cells, name='Set-'+str(j+1))
        p.SectionAssignment(region=region, sectionName='Section-1', offset=0.0, 
            offsetType=MIDDLE_SURFACE, offsetField='', 
            thicknessAssignment=FROM_SECTION)

# Assign section of matrix
for i in range(num_rve):
    p = mdb.models['Model-'+str(i+1)].parts['Part-1']
    c = p.cells
    cells = c.findAt(((- dim_rve / 2, - dim_rve / 2, 0),))
    region = p.Set(cells=cells, name='Set-'+str(num_fiber + 1))
    p.SectionAssignment(region=region, sectionName='Section-2', offset=0.0, 
        offsetType=MIDDLE_SURFACE, offsetField='', 
        thicknessAssignment=FROM_SECTION)

# Assign different colors to different materials
cmap = session.viewports['Viewport: 1'].colorMappings['Material']
session.viewports['Viewport: 1'].setColor(colorMapping=cmap)
session.viewports['Viewport: 1'].disableMultipleColors()


## 5. Generate mesh

for i in range(num_rve):
    p = mdb.models['Model-'+str(i+1)].parts['Part-1']
    # Set the sizes of the elements
    p.seedPart(size=size_mesh, deviationFactor=0.1, minSizeFactor=0.1)
    # Set the shape of the elements
    c = p.cells
    pickedRegions = c.findAt(((dim_rve / 2, dim_rve / 2, 0),))
    p.setMeshControls(regions=pickedRegions, elemShape=HEX_DOMINATED, 
        technique=SWEEP)
    p.generateMesh()


## 6. Assembly / Create steps / Amplitude / Output requests

for i in range(num_rve):
    # Assembly
    a = mdb.models['Model-'+str(i+1)].rootAssembly
    a.DatumCsysByDefault(CARTESIAN)
    p = mdb.models['Model-'+str(i+1)].parts['Part-1']
    a.Instance(name='Part-1-1', part=p, dependent=ON)
    # Create steps
    mdb.models['Model-'+str(i+1)].StaticStep(name='Step-1', previous='Initial',
        timePeriod=ini_Inc * num_time_steps, timeIncrementationMethod=FIXED, initialInc=ini_Inc, noStop=OFF)

# Create Amplitude   
for i in range(num_rve):
    
    # Preprocessing of raw dataset
    timestep_v = np.empty(num_time_steps)
    A11_i = np.zeros((num_time_steps + 1, 2))
    A22_i = np.zeros((num_time_steps + 1, 2))
    A33_i = np.zeros((num_time_steps + 1, 2))
    A12_i = np.zeros((num_time_steps + 1, 2))
    A13_i = np.zeros((num_time_steps + 1, 2))
    A23_i = np.zeros((num_time_steps + 1, 2))

    strainh11_i = strain_hat_dataset[i, :, 0]
    strainh22_i = strain_hat_dataset[i, :, 1]
    strainh33_i = strain_hat_dataset[i, :, 2]
    strainh12_i = strain_hat_dataset[i, :, 3]
    strainh13_i = strain_hat_dataset[i, :, 4]
    strainh23_i = strain_hat_dataset[i, :, 5]
    
    for j in range(num_time_steps):
        timestep_v[j] = ini_Inc * (j + 1)

    for j in range(num_time_steps):
        A11_i[j + 1][0] = timestep_v[j]
        A11_i[j + 1][1] = strainh11_i[j]
        A22_i[j + 1][0] = timestep_v[j]
        A22_i[j + 1][1] = strainh22_i[j]
        A33_i[j + 1][0] = timestep_v[j]
        A33_i[j + 1][1] = strainh33_i[j]
        A12_i[j + 1][0] = timestep_v[j]
        A12_i[j + 1][1] = strainh12_i[j]
        A13_i[j + 1][0] = timestep_v[j]
        A13_i[j + 1][1] = strainh13_i[j]
        A23_i[j + 1][0] = timestep_v[j]
        A23_i[j + 1][1] = strainh23_i[j]

    A11_i = list(A11_i)
    A22_i = list(A22_i)
    A33_i = list(A33_i)
    A12_i = list(A12_i)
    A13_i = list(A13_i)
    A23_i = list(A23_i)

    for j in range(num_time_steps + 1):
        A11_i[j] = tuple(A11_i[j])
        A22_i[j] = tuple(A22_i[j])
        A33_i[j] = tuple(A33_i[j])
        A12_i[j] = tuple(A12_i[j])
        A13_i[j] = tuple(A13_i[j])
        A23_i[j] = tuple(A23_i[j])

    A11_i = tuple(A11_i)
    A22_i = tuple(A22_i)
    A33_i = tuple(A33_i)
    A12_i = tuple(A12_i)
    A13_i = tuple(A13_i)
    A23_i = tuple(A23_i)

    # Amplitude
    mdb.models['Model-'+str(i+1)].TabularAmplitude(name='Amp-1', timeSpan=STEP, 
        smooth=SOLVER_DEFAULT, data=A11_i)
    mdb.models['Model-'+str(i+1)].TabularAmplitude(name='Amp-2', timeSpan=STEP, 
        smooth=SOLVER_DEFAULT, data=A22_i)
    mdb.models['Model-'+str(i+1)].TabularAmplitude(name='Amp-3', timeSpan=STEP, 
        smooth=SOLVER_DEFAULT, data=A33_i)
    mdb.models['Model-'+str(i+1)].TabularAmplitude(name='Amp-4', timeSpan=STEP, 
        smooth=SOLVER_DEFAULT, data=A12_i)
    mdb.models['Model-'+str(i+1)].TabularAmplitude(name='Amp-5', timeSpan=STEP, 
        smooth=SOLVER_DEFAULT, data=A13_i)
    mdb.models['Model-'+str(i+1)].TabularAmplitude(name='Amp-6', timeSpan=STEP, 
        smooth=SOLVER_DEFAULT, data=A23_i)
    
Amp_name_v = ['Amp-1', 'Amp-4', 'Amp-5', 'Amp-4', 'Amp-2', 'Amp-6', 'Amp-5', 'Amp-6', 'Amp-3']

# Output requests
for i in range(num_rve):
    mdb.models['Model-'+str(i+1)].fieldOutputRequests['F-Output-1'].setValues(variables=(
    'S', 'PE', 'PEEQ', 'PEMAG', 'LE', 'U', 'RF', 'CF', 'CSTRESS', 'CDISP', 
    'SVOL', 'EVOL', 'ESOL', 'IVOL', 'STH', 'COORD'))


## 7. Apply Boundary Conditions

# 7.1 Generate reference points / Apply BC for RPs

# Position of RPs
RP_v = [[dim_rve, 0, dim_rve/2], [3 * dim_rve/2, 0, dim_rve/2], [2 * dim_rve, 0, dim_rve/2],
        [0, dim_rve, dim_rve/2], [0, 3 * dim_rve/2, dim_rve/2], [0, 2 * dim_rve, dim_rve/2],
        [0, 0, 3 * dim_rve/2], [0, 0, 2 * dim_rve], [0, 0, 5 * dim_rve/2]]

# Generate reference points
for i in range(num_rve):
    for j in range(9):
        a = mdb.models['Model-'+str(i+1)].rootAssembly
        a.ReferencePoint(point=(RP_v[j][0], RP_v[j][1], RP_v[j][2]))
        
# Apply BC for RPs
for i in range(num_rve):
    for j in range(3):
        a = mdb.models['Model-'+str(i+1)].rootAssembly
        r1 = a.referencePoints
        # Apply BC in X direction
        refPoints1=(r1[4 + 3*j], )
        region = a.Set(referencePoints=refPoints1, name='Set-RP-'+str(3*j+1))
        mdb.models['Model-'+str(i+1)].DisplacementBC(name='BC-RP-'+str(3*j+1), createStepName='Step-1', 
            region=region, u1=1.0, u2=UNSET, u3=UNSET, ur1=UNSET, ur2=UNSET, ur3=UNSET, 
            amplitude=Amp_name_v[j], fixed=OFF, distributionType=UNIFORM, fieldName='', 
            localCsys=None)
        # Apply BC in Y direction
        refPoints1=(r1[5 + 3*j], )
        region = a.Set(referencePoints=refPoints1, name='Set-RP-'+str(3*j+2))
        mdb.models['Model-'+str(i+1)].DisplacementBC(name='BC-RP-'+str(3*j+2), createStepName='Step-1', 
            region=region, u1=UNSET, u2=1.0, u3=UNSET, ur1=UNSET, ur2=UNSET, ur3=UNSET, 
            amplitude=Amp_name_v[j+3], fixed=OFF, distributionType=UNIFORM, fieldName='', 
            localCsys=None)
        # Apply BC in Z direction
        refPoints1=(r1[6 + 3*j], )
        region = a.Set(referencePoints=refPoints1, name='Set-RP-'+str(3*j+3))
        mdb.models['Model-'+str(i+1)].DisplacementBC(name='BC-RP-'+str(3*j+3), createStepName='Step-1', 
            region=region, u1=UNSET, u2=UNSET, u3=1.0, ur1=UNSET, ur2=UNSET, ur3=UNSET, 
            amplitude=Amp_name_v[j+6], fixed=OFF, distributionType=UNIFORM, fieldName='', 
            localCsys=None)

# 7.2 Load strain data
eps_hat_v = [1, 1, 1, 1, 1, 1]

# 7.3 Apply PBC

for i_pbc in range(num_rve):
    
    # 7.3.1 Preparation

    modelName = 'Model-'+str(i_pbc+1)
    instanceName = 'Part-1-1'
    Nodeset = mdb.models[modelName].rootAssembly.instances[instanceName].nodes

    j = 0
    x = []
    y = []
    z = []
    Nodeset_surface = []
    x_p = []
    x_n = []
    y_p = []
    y_n = []
    z_p = []
    z_n = []
    edge_15 = []
    edge_37 = []
    edge_48 = []
    edge_26 = []
    edge_78 = []
    edge_34 = []
    edge_12 = []
    edge_56 = []
    edge_13 = []
    edge_57 = []
    edge_68 = []
    edge_24 = []

    # Identifying RVE size
    for i in Nodeset:
        x.insert(j,i.coordinates[0])
        y.insert(j,i.coordinates[1])
        z.insert(j,i.coordinates[2])
        j=j+1

    Max = max(x)
    May = max(y)
    Maz = max(z)
    Mnx = min(x)
    Mny = min(y)
    Mnz = min(z)

    #print(Nodeset[0])
    #print(Max)

    # 7.3.2 Change data structure and create sets

    # Change data structure and create node sets of entire surface
    for i in Nodeset:
        if i.coordinates[0] == Max or i.coordinates[0] == Mnx or \
            i.coordinates[1] == May or i.coordinates[1] == Mny or \
            i.coordinates[2] == Maz or i.coordinates[2] == Mnz:
            node = (i.label, i.coordinates[0], i.coordinates[1], i.coordinates[2], i)
            Nodeset_surface.append(node)

    # Create node sets of each surface
    for i in Nodeset_surface:
        if i[1] == Max and i[2] < May and i[2] > Mny and \
            i[3] < Maz and i[3] > Mnz:
            x_p.append(i)
        elif i[1] == Mnx and i[2] < May and i[2] > Mny and \
            i[3] < Maz and i[3] > Mnz:
            x_n.append(i)
    for i in Nodeset_surface:
        if i[2] == May and i[1] < Max and i[1] > Mnx and \
            i[3] < Maz and i[3] > Mnz:
            y_p.append(i)
        elif i[2] == Mny and i[1] < Max and i[1] > Mnx and \
            i[3] < Maz and i[3] > Mnz:
            y_n.append(i)
    for i in Nodeset_surface:
        if i[3] == Maz and i[1] < Max and i[1] > Mnx and \
            i[2] < May and i[2] > Mny:
            z_p.append(i)
        elif i[3] == Mnz and i[1] < Max and i[1] > Mnx and \
            i[2] < May and i[2] > Mny:
            z_n.append(i)

    # Create node sets of each edge
    for i in Nodeset_surface:
        if i[2] == Mny and i[3] == Maz and i[1] < Max and i[1] > Mnx:
            edge_15.append(i)
        elif i[2] == Mny and i[3] == Mnz and i[1] < Max and i[1] > Mnx:
            edge_37.append(i)
        elif i[2] == May and i[3] == Mnz and i[1] < Max and i[1] > Mnx:
            edge_48.append(i)
        elif i[2] == May and i[3] == Maz and i[1] < Max and i[1] > Mnx:
            edge_26.append(i)
        elif i[1] == Max and i[3] == Mnz and i[2] < May and i[2] > Mny:
            edge_78.append(i)
        elif i[1] == Mnx and i[3] == Mnz and i[2] < May and i[2] > Mny:
            edge_34.append(i)
        elif i[1] == Mnx and i[3] == Maz and i[2] < May and i[2] > Mny:
            edge_12.append(i)
        elif i[1] == Max and i[3] == Maz and i[2] < May and i[2] > Mny:
            edge_56.append(i)
        elif i[1] == Mnx and i[2] == Mny and i[3] < Maz and i[3] > Mnz:
            edge_13.append(i)
        elif i[1] == Max and i[2] == Mny and i[3] < Maz and i[3] > Mnz:
            edge_57.append(i)
        elif i[1] == Max and i[2] == May and i[3] < Maz and i[3] > Mnz:
            edge_68.append(i)
        elif i[1] == Mnx and i[2] == May and i[3] < Maz and i[3] > Mnz:
            edge_24.append(i)

    # Save the length of each set
    x_p_l = len(x_p)
    x_n_l = len(x_n)
    y_p_l = len(y_p)
    y_n_l = len(y_n)
    z_p_l = len(z_p)
    z_n_l = len(z_n)
    edge_15_l = len(edge_15)
    edge_37_l = len(edge_37)
    edge_48_l = len(edge_48)
    edge_26_l = len(edge_26)
    edge_78_l = len(edge_78)
    edge_34_l = len(edge_34)
    edge_12_l = len(edge_12)
    edge_56_l = len(edge_56)
    edge_13_l = len(edge_13)
    edge_57_l = len(edge_57)
    edge_68_l = len(edge_68)
    edge_24_l = len(edge_24)

    # Create node sets of each nodes in the surface
    Nodeset_surface_l = len(Nodeset_surface)

    for i in range(Nodeset_surface_l):
        a = mdb.models['Model-'+str(i_pbc+1)].rootAssembly
        label = [Nodeset_surface[i][0]]
        label = tuple(label)
        a.SetFromNodeLabels(name='Set-pbc-'+str(Nodeset_surface[i][0]), nodeLabels=(('Part-1-1',label),))

    # Assign set names of the nodes in the vertices to variable
    for i in Nodeset_surface:
        if i[1] == Mnx and i[2] == Mny and i[3] == Maz:
            node_set_vertex_1 = 'Set-pbc-'+str(i[0])
        if i[1] == Mnx and i[2] == May and i[3] == Maz:
            node_set_vertex_2 = 'Set-pbc-'+str(i[0])
        if i[1] == Mnx and i[2] == Mny and i[3] == Mnz:
            node_set_vertex_3 = 'Set-pbc-'+str(i[0])
        if i[1] == Mnx and i[2] == May and i[3] == Mnz:
            node_set_vertex_4 = 'Set-pbc-'+str(i[0])
        if i[1] == Max and i[2] == Mny and i[3] == Maz:
            node_set_vertex_5 = 'Set-pbc-'+str(i[0])
        if i[1] == Max and i[2] == May and i[3] == Maz:
            node_set_vertex_6 = 'Set-pbc-'+str(i[0])
        if i[1] == Max and i[2] == Mny and i[3] == Mnz:
            node_set_vertex_7 = 'Set-pbc-'+str(i[0])
        if i[1] == Max and i[2] == May and i[3] == Mnz:
            node_set_vertex_8 = 'Set-pbc-'+str(i[0])

    # 7.3.3 Create constraints

    # Assign set names of reference points to variable
    RP_1_set = 'Set-RP-1'
    RP_2_set = 'Set-RP-2'
    RP_3_set = 'Set-RP-3'
    RP_4_set = 'Set-RP-4'
    RP_5_set = 'Set-RP-5'
    RP_6_set = 'Set-RP-6'
    RP_7_set = 'Set-RP-7'
    RP_8_set = 'Set-RP-8'
    RP_9_set = 'Set-RP-9'

    # Create constraints of the nodes in the pure surfaces
    
    for i in range(x_p_l):
        for j in range(x_n_l):
            if abs(x_p[i][2] - x_n[j][2]) <= 0.0000001 and abs(x_p[i][3] - x_n[j][3]) <= 0.0000001:
                node_xp = x_p[i][0]
                node_xn = x_n[j][0]
                node_set_xp = 'Set-pbc-'+str(node_xp)
                node_set_xn = 'Set-pbc-'+str(node_xn)
                mdb.models['Model-'+str(i_pbc+1)].Equation(name='Constraint-x-x-'+str(node_xp), 
                                                           terms=((1.0, node_set_xp, 1), (-1.0, node_set_xn, 1), (-eps_hat_v[0], RP_1_set, 1)))
                mdb.models['Model-'+str(i_pbc+1)].Equation(name='Constraint-x-y-'+str(node_xp), 
                                                           terms=((1.0, node_set_xp, 2), (-1.0, node_set_xn, 2), (-eps_hat_v[3], RP_2_set, 2)))
                mdb.models['Model-'+str(i_pbc+1)].Equation(name='Constraint-x-z-'+str(node_xp), 
                                                           terms=((1.0, node_set_xp, 3), (-1.0, node_set_xn, 3), (-eps_hat_v[4], RP_3_set, 3)))

    for i in range(y_p_l):
        for j in range(y_n_l):
            if abs(y_p[i][1] - y_n[j][1]) <= 0.0000001 and abs(y_p[i][3] - y_n[j][3]) <= 0.0000001:
                node_yp = y_p[i][0]
                node_yn = y_n[j][0]
                node_set_yp = 'Set-pbc-'+str(node_yp)
                node_set_yn = 'Set-pbc-'+str(node_yn)
                mdb.models['Model-'+str(i_pbc+1)].Equation(name='Constraint-y-x-'+str(node_yp), 
                                                           terms=((1.0, node_set_yp, 1), (-1.0, node_set_yn, 1), (-0, RP_4_set, 1)))
                mdb.models['Model-'+str(i_pbc+1)].Equation(name='Constraint-y-y-'+str(node_yp), 
                                                           terms=((1.0, node_set_yp, 2), (-1.0, node_set_yn, 2), (-eps_hat_v[1], RP_5_set, 2)))
                mdb.models['Model-'+str(i_pbc+1)].Equation(name='Constraint-y-z-'+str(node_yp), 
                                                           terms=((1.0, node_set_yp, 3), (-1.0, node_set_yn, 3), (-eps_hat_v[5], RP_6_set, 3)))

    for i in range(z_p_l):
        for j in range(z_n_l):
            if abs(z_p[i][1] - z_n[j][1]) <= 0.0000001 and abs(z_p[i][2] - z_n[j][2]) <= 0.0000001:
                node_zp = z_p[i][0]
                node_zn = z_n[j][0]
                node_set_zp = 'Set-pbc-'+str(node_zp)
                node_set_zn = 'Set-pbc-'+str(node_zn)
                mdb.models['Model-'+str(i_pbc+1)].Equation(name='Constraint-z-x-'+str(node_zp), 
                                                           terms=((1.0, node_set_zp, 1), (-1.0, node_set_zn, 1), (-0, RP_7_set, 1)))
                mdb.models['Model-'+str(i_pbc+1)].Equation(name='Constraint-z-y-'+str(node_zp), 
                                                           terms=((1.0, node_set_zp, 2), (-1.0, node_set_zn, 2), (-0, RP_8_set, 2)))
                mdb.models['Model-'+str(i_pbc+1)].Equation(name='Constraint-z-z-'+str(node_zp), 
                                                           terms=((1.0, node_set_zp, 3), (-1.0, node_set_zn, 3), (-eps_hat_v[2], RP_9_set, 3)))
                
    # Create constraints of the nodes in the edges

    for i in range(edge_15_l):
        for j in range(edge_37_l):
            if abs(edge_15[i][1] - edge_37[j][1]) <= 0.0000001:
                node_1 = edge_15[i][0]
                node_2 = edge_37[j][0]
                node_set_1 = 'Set-pbc-'+str(node_1)
                node_set_2 = 'Set-pbc-'+str(node_2)
                mdb.models['Model-'+str(i_pbc+1)].Equation(name='Constraint-15-37-x-'+str(node_1), 
                                                           terms=((1.0, node_set_1, 1), (-1.0, node_set_2, 1), (-0, RP_7_set, 1)))
                mdb.models['Model-'+str(i_pbc+1)].Equation(name='Constraint-15-37-y-'+str(node_1), 
                                                           terms=((1.0, node_set_1, 2), (-1.0, node_set_2, 2), (-0, RP_8_set, 2)))
                mdb.models['Model-'+str(i_pbc+1)].Equation(name='Constraint-15-37-z-'+str(node_1), 
                                                           terms=((1.0, node_set_1, 3), (-1.0, node_set_2, 3), (-eps_hat_v[2], RP_9_set, 3)))
                
    for i in range(edge_37_l):
        for j in range(edge_48_l):
            if abs(edge_37[i][1] - edge_48[j][1]) <= 0.0000001:
                node_1 = edge_37[i][0]
                node_2 = edge_48[j][0]
                node_set_1 = 'Set-pbc-'+str(node_1)
                node_set_2 = 'Set-pbc-'+str(node_2)
                mdb.models['Model-'+str(i_pbc+1)].Equation(name='Constraint-37-48-x-'+str(node_1), 
                                                           terms=((1.0, node_set_1, 1), (-1.0, node_set_2, 1), (0, RP_4_set, 1)))
                mdb.models['Model-'+str(i_pbc+1)].Equation(name='Constraint-37-48-y-'+str(node_1), 
                                                           terms=((1.0, node_set_1, 2), (-1.0, node_set_2, 2), (eps_hat_v[1], RP_5_set, 2)))
                mdb.models['Model-'+str(i_pbc+1)].Equation(name='Constraint-37-48-z-'+str(node_1), 
                                                           terms=((1.0, node_set_1, 3), (-1.0, node_set_2, 3), (eps_hat_v[5], RP_6_set, 3)))
                
    for i in range(edge_48_l):
        for j in range(edge_26_l):
            if abs(edge_48[i][1] - edge_26[j][1]) <= 0.0000001:
                node_1 = edge_48[i][0]
                node_2 = edge_26[j][0]
                node_set_1 = 'Set-pbc-'+str(node_1)
                node_set_2 = 'Set-pbc-'+str(node_2)
                mdb.models['Model-'+str(i_pbc+1)].Equation(name='Constraint-48-26-x-'+str(node_1), 
                                                           terms=((1.0, node_set_1, 1), (-1.0, node_set_2, 1), (0, RP_7_set, 1)))
                mdb.models['Model-'+str(i_pbc+1)].Equation(name='Constraint-48-26-y-'+str(node_1), 
                                                           terms=((1.0, node_set_1, 2), (-1.0, node_set_2, 2), (0, RP_8_set, 2)))
                mdb.models['Model-'+str(i_pbc+1)].Equation(name='Constraint-48-26-z-'+str(node_1), 
                                                           terms=((1.0, node_set_1, 3), (-1.0, node_set_2, 3), (eps_hat_v[2], RP_9_set, 3)))
                
    for i in range(edge_78_l):
        for j in range(edge_34_l):
            if abs(edge_78[i][2] - edge_34[j][2]) <= 0.0000001:
                node_1 = edge_78[i][0]
                node_2 = edge_34[j][0]
                node_set_1 = 'Set-pbc-'+str(node_1)
                node_set_2 = 'Set-pbc-'+str(node_2)
                mdb.models['Model-'+str(i_pbc+1)].Equation(name='Constraint-78-34-x-'+str(node_1), 
                                                           terms=((1.0, node_set_1, 1), (-1.0, node_set_2, 1), (-eps_hat_v[0], RP_1_set, 1)))
                mdb.models['Model-'+str(i_pbc+1)].Equation(name='Constraint-78-34-y-'+str(node_1), 
                                                           terms=((1.0, node_set_1, 2), (-1.0, node_set_2, 2), (-eps_hat_v[3], RP_2_set, 2)))
                mdb.models['Model-'+str(i_pbc+1)].Equation(name='Constraint-78-34-z-'+str(node_1), 
                                                           terms=((1.0, node_set_1, 3), (-1.0, node_set_2, 3), (-eps_hat_v[4], RP_3_set, 3)))
                
    for i in range(edge_34_l):
        for j in range(edge_12_l):
            if abs(edge_34[i][2] - edge_12[j][2]) <= 0.0000001:
                node_1 = edge_34[i][0]
                node_2 = edge_12[j][0]
                node_set_1 = 'Set-pbc-'+str(node_1)
                node_set_2 = 'Set-pbc-'+str(node_2)
                mdb.models['Model-'+str(i_pbc+1)].Equation(name='Constraint-34-12-x-'+str(node_1), 
                                                           terms=((1.0, node_set_1, 1), (-1.0, node_set_2, 1), (0, RP_7_set, 1)))
                mdb.models['Model-'+str(i_pbc+1)].Equation(name='Constraint-34-12-y-'+str(node_1), 
                                                           terms=((1.0, node_set_1, 2), (-1.0, node_set_2, 2), (0, RP_8_set, 2)))
                mdb.models['Model-'+str(i_pbc+1)].Equation(name='Constraint-34-12-z-'+str(node_1), 
                                                           terms=((1.0, node_set_1, 3), (-1.0, node_set_2, 3), (eps_hat_v[2], RP_9_set, 3)))
                
    for i in range(edge_12_l):
        for j in range(edge_56_l):
            if abs(edge_12[i][2] - edge_56[j][2]) <= 0.0000001:
                node_1 = edge_12[i][0]
                node_2 = edge_56[j][0]
                node_set_1 = 'Set-pbc-'+str(node_1)
                node_set_2 = 'Set-pbc-'+str(node_2)
                mdb.models['Model-'+str(i_pbc+1)].Equation(name='Constraint-12-56-x-'+str(node_1), 
                                                           terms=((1.0, node_set_1, 1), (-1.0, node_set_2, 1), (eps_hat_v[0], RP_1_set, 1)))
                mdb.models['Model-'+str(i_pbc+1)].Equation(name='Constraint-12-56-y-'+str(node_1), 
                                                           terms=((1.0, node_set_1, 2), (-1.0, node_set_2, 2), (eps_hat_v[3], RP_2_set, 2)))
                mdb.models['Model-'+str(i_pbc+1)].Equation(name='Constraint-12-56-z-'+str(node_1), 
                                                           terms=((1.0, node_set_1, 3), (-1.0, node_set_2, 3), (eps_hat_v[4], RP_3_set, 3)))
                
    for i in range(edge_13_l):
        for j in range(edge_57_l):
            if abs(edge_13[i][3] - edge_57[j][3]) <= 0.0000001:
                node_1 = edge_13[i][0]
                node_2 = edge_57[j][0]
                node_set_1 = 'Set-pbc-'+str(node_1)
                node_set_2 = 'Set-pbc-'+str(node_2)
                mdb.models['Model-'+str(i_pbc+1)].Equation(name='Constraint-13-57-x-'+str(node_1), 
                                                           terms=((1.0, node_set_1, 1), (-1.0, node_set_2, 1), (eps_hat_v[0], RP_1_set, 1)))
                mdb.models['Model-'+str(i_pbc+1)].Equation(name='Constraint-13-57-y-'+str(node_1), 
                                                           terms=((1.0, node_set_1, 2), (-1.0, node_set_2, 2), (eps_hat_v[3], RP_2_set, 2)))
                mdb.models['Model-'+str(i_pbc+1)].Equation(name='Constraint-13-57-z-'+str(node_1), 
                                                           terms=((1.0, node_set_1, 3), (-1.0, node_set_2, 3), (eps_hat_v[4], RP_3_set, 3)))
                
    for i in range(edge_57_l):
        for j in range(edge_68_l):
            if abs(edge_57[i][3] - edge_68[j][3]) <= 0.0000001:
                node_1 = edge_57[i][0]
                node_2 = edge_68[j][0]
                node_set_1 = 'Set-pbc-'+str(node_1)
                node_set_2 = 'Set-pbc-'+str(node_2)
                mdb.models['Model-'+str(i_pbc+1)].Equation(name='Constraint-57-68-x-'+str(node_1), 
                                                           terms=((1.0, node_set_1, 1), (-1.0, node_set_2, 1), (0, RP_4_set, 1)))
                mdb.models['Model-'+str(i_pbc+1)].Equation(name='Constraint-57-68-y-'+str(node_1), 
                                                           terms=((1.0, node_set_1, 2), (-1.0, node_set_2, 2), (eps_hat_v[1], RP_5_set, 2)))
                mdb.models['Model-'+str(i_pbc+1)].Equation(name='Constraint-57-68-z-'+str(node_1), 
                                                           terms=((1.0, node_set_1, 3), (-1.0, node_set_2, 3), (eps_hat_v[5], RP_6_set, 3)))
                
    for i in range(edge_68_l):
        for j in range(edge_24_l):
            if abs(edge_68[i][3] - edge_24[j][3]) <= 0.0000001:
                node_1 = edge_68[i][0]
                node_2 = edge_24[j][0]
                node_set_1 = 'Set-pbc-'+str(node_1)
                node_set_2 = 'Set-pbc-'+str(node_2)
                mdb.models['Model-'+str(i_pbc+1)].Equation(name='Constraint-68-24-x-'+str(node_1), 
                                                           terms=((1.0, node_set_1, 1), (-1.0, node_set_2, 1), (-eps_hat_v[0], RP_1_set, 1)))
                mdb.models['Model-'+str(i_pbc+1)].Equation(name='Constraint-68-24-y-'+str(node_1), 
                                                           terms=((1.0, node_set_1, 2), (-1.0, node_set_2, 2), (-eps_hat_v[3], RP_2_set, 2)))
                mdb.models['Model-'+str(i_pbc+1)].Equation(name='Constraint-68-24-z-'+str(node_1), 
                                                           terms=((1.0, node_set_1, 3), (-1.0, node_set_2, 3), (-eps_hat_v[4], RP_3_set, 3)))
                
    # Create constraints of the nodes in the vertices

    # Vertex 5 to 1
    mdb.models['Model-'+str(i_pbc+1)].Equation(name='Constraint-5-1-x', 
                                               terms=((1.0, node_set_vertex_5, 1), (-1.0, node_set_vertex_1, 1), (-eps_hat_v[0], RP_1_set, 1)))
    mdb.models['Model-'+str(i_pbc+1)].Equation(name='Constraint-5-1-y', 
                                               terms=((1.0, node_set_vertex_5, 2), (-1.0, node_set_vertex_1, 2), (-eps_hat_v[3], RP_2_set, 2)))
    mdb.models['Model-'+str(i_pbc+1)].Equation(name='Constraint-5-1-z', 
                                               terms=((1.0, node_set_vertex_5, 3), (-1.0, node_set_vertex_1, 3), (-eps_hat_v[4], RP_3_set, 3)))
    # Vertex 1 to 2
    mdb.models['Model-'+str(i_pbc+1)].Equation(name='Constraint-1-2-x', 
                                               terms=((1.0, node_set_vertex_1, 1), (-1.0, node_set_vertex_2, 1), (0, RP_4_set, 1)))
    mdb.models['Model-'+str(i_pbc+1)].Equation(name='Constraint-1-2-y', 
                                               terms=((1.0, node_set_vertex_1, 2), (-1.0, node_set_vertex_2, 2), (eps_hat_v[1], RP_5_set, 2)))
    mdb.models['Model-'+str(i_pbc+1)].Equation(name='Constraint-1-2-z', 
                                               terms=((1.0, node_set_vertex_1, 3), (-1.0, node_set_vertex_2, 3), (eps_hat_v[5], RP_6_set, 3)))
    # Vertex 2 to 6
    mdb.models['Model-'+str(i_pbc+1)].Equation(name='Constraint-2-6-x', 
                                               terms=((1.0, node_set_vertex_2, 1), (-1.0, node_set_vertex_6, 1), (eps_hat_v[0], RP_1_set, 1)))
    mdb.models['Model-'+str(i_pbc+1)].Equation(name='Constraint-2-6-y', 
                                               terms=((1.0, node_set_vertex_2, 2), (-1.0, node_set_vertex_6, 2), (eps_hat_v[3], RP_2_set, 2)))
    mdb.models['Model-'+str(i_pbc+1)].Equation(name='Constraint-2-6-z', 
                                               terms=((1.0, node_set_vertex_2, 3), (-1.0, node_set_vertex_6, 3), (eps_hat_v[4], RP_3_set, 3)))
    # Vertex 6 to 8
    mdb.models['Model-'+str(i_pbc+1)].Equation(name='Constraint-6-8-x', 
                                               terms=((1.0, node_set_vertex_6, 1), (-1.0, node_set_vertex_8, 1), (-0, RP_7_set, 1)))
    mdb.models['Model-'+str(i_pbc+1)].Equation(name='Constraint-6-8-y', 
                                               terms=((1.0, node_set_vertex_6, 2), (-1.0, node_set_vertex_8, 2), (-0, RP_8_set, 2)))
    mdb.models['Model-'+str(i_pbc+1)].Equation(name='Constraint-6-8-z', 
                                               terms=((1.0, node_set_vertex_6, 3), (-1.0, node_set_vertex_8, 3), (-eps_hat_v[2], RP_9_set, 3)))
    # Vertex 8 to 4
    mdb.models['Model-'+str(i_pbc+1)].Equation(name='Constraint-8-4-x', 
                                               terms=((1.0, node_set_vertex_8, 1), (-1.0, node_set_vertex_4, 1), (-eps_hat_v[0], RP_1_set, 1)))
    mdb.models['Model-'+str(i_pbc+1)].Equation(name='Constraint-8-4-y', 
                                               terms=((1.0, node_set_vertex_8, 2), (-1.0, node_set_vertex_4, 2), (-eps_hat_v[3], RP_2_set, 2)))
    mdb.models['Model-'+str(i_pbc+1)].Equation(name='Constraint-8-4-z', 
                                               terms=((1.0, node_set_vertex_8, 3), (-1.0, node_set_vertex_4, 3), (-eps_hat_v[4], RP_3_set, 3)))
    # Vertex 4 to 3
    mdb.models['Model-'+str(i_pbc+1)].Equation(name='Constraint-4-3-x', 
                                               terms=((1.0, node_set_vertex_4, 1), (-1.0, node_set_vertex_3, 1), (-0, RP_4_set, 1)))
    mdb.models['Model-'+str(i_pbc+1)].Equation(name='Constraint-4-3-y', 
                                               terms=((1.0, node_set_vertex_4, 2), (-1.0, node_set_vertex_3, 2), (-eps_hat_v[1], RP_5_set, 2)))
    mdb.models['Model-'+str(i_pbc+1)].Equation(name='Constraint-4-3-z', 
                                               terms=((1.0, node_set_vertex_4, 3), (-1.0, node_set_vertex_3, 3), (-eps_hat_v[5], RP_6_set, 3)))
    # Vertex 3 to 7
    mdb.models['Model-'+str(i_pbc+1)].Equation(name='Constraint-3-7-x', 
                                               terms=((1.0, node_set_vertex_3, 1), (-1.0, node_set_vertex_7, 1), (eps_hat_v[0], RP_1_set, 1)))
    mdb.models['Model-'+str(i_pbc+1)].Equation(name='Constraint-3-7-y', 
                                               terms=((1.0, node_set_vertex_3, 2), (-1.0, node_set_vertex_7, 2), (eps_hat_v[3], RP_2_set, 2)))
    mdb.models['Model-'+str(i_pbc+1)].Equation(name='Constraint-3-7-z', 
                                               terms=((1.0, node_set_vertex_3, 3), (-1.0, node_set_vertex_7, 3), (eps_hat_v[4], RP_3_set, 3)))
    
    print(i_pbc+1)


## 8. Assign loads

# We do not need this section in this case
        

## 9. Create jobs / write .inp file

for i in range(num_rve):
    a = mdb.models['Model-'+str(i+1)].rootAssembly
    mdb.Job(name='Job-MSThesis-FixF1-'+str(i+1), model='Model-'+str(i+1), description='', type=ANALYSIS, 
        atTime=None, waitMinutes=0, waitHours=0, queue=None, memory=90, 
        memoryUnits=PERCENTAGE, getMemoryFromAnalysis=True, 
        explicitPrecision=SINGLE, nodalOutputPrecision=SINGLE, echoPrint=OFF, 
        modelPrint=OFF, contactPrint=OFF, historyPrint=OFF, userSubroutine='', 
        scratch='', resultsFormat=ODB, numThreadsPerMpiProcess=1, 
        multiprocessingMode=DEFAULT, numCpus=1, numGPUs=0)
    
for i in range(num_rve):
    mdb.jobs['Job-MSThesis-FixF1-'+str(i+1)].writeInput(consistencyChecking=OFF)

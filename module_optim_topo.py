# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 11:30:20 2023

@author: cherriere
"""

# Module optim topo

from ngsolve import *
from ngsolve.webgui import Draw
from netgen.geom2d import CSG2d, Circle, Rectangle
from copy import copy
import numpy as np
import matplotlib.pyplot as plt


# Maillage disque laminé

def meshLamDisk(Nbar, h = 1):
    geo = CSG2d()
    R=1
    x = R*2*(np.append(np.insert(np.arange(0.5,Nbar+0.5),0,0),Nbar)/Nbar-0.5)
    
    circle1 = Circle( center=(0,0), radius=R, bc="left_up" ) * Rectangle( pmin=(-R,0), pmax=(0,R))
    circle2 = Circle( center=(0,0), radius=R, bc="left_bot" ) * Rectangle( pmin=(-R,-R), pmax=(0,0))
    circle3 = Circle( center=(0,0), radius=R, bc="right_bot" ) * Rectangle( pmin=(0,-R), pmax=(R,0))
    circle4 = Circle( center=(0,0), radius=R, bc="right_up" ) * Rectangle( pmin=(0,0), pmax=(R,R))
    
    materials = ["iron","air"]
    
    for i in range(len(x)-1):
        geo.Add(Rectangle( pmin=(x[i],-R), pmax=(x[i+1],R), mat = materials[i%2] ) * (circle1 + circle2 + circle3 +circle4))

    # Attention à fixer la taille du maillage sinon le volume change à cause des elts grossiers
    m = geo.GenerateMesh(maxh=h)
    return Mesh(m)


def solvePrimal_linear(mur,beta,mesh):
    # le champ 1 est vertical, le champ 2 est horizontal
    # on impose les forces magnétomotrices
    
    nu0 = 4e-7*np.pi
    nu = mesh.MaterialCF({ "iron" : 1/(mu0*mur) }, default=1/mu0)
    
    fespace_H1 = H1(mesh, order=1)
    fespace_H1.FreeDofs()[0] = False
    a = fespace_H1.TrialFunction()
    psi = fespace_H1.TestFunction()
    K = BilinearForm(fespace_H1, symmetric=True)
    K +=  grad(psi)* nu *grad(a) *dx
    
    # imposition des circulations sur le côté gauche et droit
    l1 = LinearForm(fespace_H1)
    l1 += -psi*beta*sqrt(1-y*y)*ds(definedon=mesh.Boundaries("right_bot|right_up"))
    l1 += psi*beta*sqrt(1-y*y)*ds(definedon=mesh.Boundaries("left_bot|left_up"))
    
    # imposition des circulations sur le haut et le bas
    l2 = LinearForm(fespace_H1)
    l2 += -psi* beta * sqrt(1-x*x)* ds(definedon=mesh.Boundaries("right_bot|left_bot"))
    l2 += psi*beta* sqrt(1-x*x)*ds(definedon=mesh.Boundaries("right_up|left_up"))

    K.Assemble()
    Kdec = K.mat.Inverse(inverse="sparsecholesky")
    
    a1 = GridFunction(fespace_H1)  # solution
    a1.vec.data =     Kdec * l1.Assemble().vec
    a2 = GridFunction(fespace_H1)  # solution
    a2.vec.data =     Kdec * l2.Assemble().vec
    
    return a1, a2

def solveDual_linear(mur,beta,mesh):
    # le champ 1 est vertical, le champ 2 est horizontal
    # on impose les flux
    
    mu0 = 4e-7*np.pi
    mu = mesh.MaterialCF({ "iron" : mur*mu0 }, default=mu0)
    
    fespace_H1 = H1(mesh, order=1)
    fespace_H1.FreeDofs()[0] = False
    phi = fespace_H1.TrialFunction()
    psi = fespace_H1.TestFunction()
    K = BilinearForm(fespace_H1, symmetric=True)
    K +=  grad(psi)* nu *grad(phi) *dx
    
    # imposition du flux sur le côté haut et bas
    l1 = LinearForm(fespace_H1)
    l1 += -psi* beta * sqrt(1-x*x)* ds(definedon=mesh.Boundaries("right_bot|left_bot"))
    l1 += psi*beta* sqrt(1-x*x)*ds(definedon=mesh.Boundaries("right_up|left_up"))
    
    
    # imposition du flux sur le côté gauche et droit
    l2 = LinearForm(fespace_H1)
    l2 += -psi*beta*sqrt(1-y*y)*ds(definedon=mesh.Boundaries("right_bot|right_up"))
    l2 += psi*beta*sqrt(1-y*y)*ds(definedon=mesh.Boundaries("left_bot|left_up"))

    K.Assemble()
    Kdec = K.mat.Inverse(inverse="sparsecholesky")
    
    phi1 = GridFunction(fespace_H1)  # solution
    phi1.vec.data =     Kdec * l1.Assemble().vec
    phi2 = GridFunction(fespace_H1)  # solution
    phi2.vec.data =     Kdec * l2.Assemble().vec
    
    return phi1, phi2

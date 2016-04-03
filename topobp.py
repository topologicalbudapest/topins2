from numpy import *
import scipy.sparse as scsp
import scipy.sparse.linalg as scspl

#Define the Pauli matrices unit matrix and the zero matrix
s0=matrix([[1,0],[0,1]])
s1=matrix([[0,1],[1,0]])
s2=matrix([[0,-1j],[1j,0]])
s3=matrix([[1,0],[0,-1]])
z2=zeros_like(s0);

## Usful many-body operators
   
def fermion_Fock_matrices(NN=3,dense=False,**kwargs):
    '''
    Returns list of 2^NN X 2^NN sparse matrices, 
    representing fermionic annihilation operators
    acting on the Fock space of NN fermions.
    Creation perators are the hermitian conjugate
    of the annihilation operators, e.g.:
    cc[0].H is the creation operator acting on the
    0th degree of freedom.
    If dense=True keyword argument is given
    then a dense representation of the operators
    is returned.
    The binary sequence Fock space representation
    is used, that is for the case of 3 degrees of freedom
    000,001,010,011,100,101,110,111 is the basis.
    '''
    
    
    l=list(map(lambda x: list(map(int,list(binary_repr(x,NN)))),arange(0,2**NN)))
    ll=-(-1)**cumsum(l,axis=1)
    AA=(array(l)*array(ll))[:,::-1]
    
    cc=[]
    for p in range(NN):
        cc.append(scsp.dia_matrix((AA[:,p],array([2**p])),shape=(2**NN,2**NN), dtype='d'))

    if dense:
        for i in range(NN):
            cc[i]=cc[i].todense()
            
    return cc


def parity_Fock_operator(NN=3,dense=False,**kwargs):
    '''
    Returns particle number parity operator in Fock
    space of NN fermions.
    If dense=True keyword argument is given 
    then a dense representation of the operators
    is returned.
    '''
    
    # list containing the diagonal elements of the parity operator
    par_diag=list(map(lambda x:1-2*mod(bin(x).count("1"),2),arange(0,2**NN)))
    # creating a sparse matrix repersenting the parity operator
    PAR=scsp.dia_matrix((par_diag,[0]),shape=(2**NN,2**NN))
    
    if dense:
        return PAR.todense()
    else:
        return PAR


def even_odd_Fock_operators(NN=3,dense=False,**kwargs):
    '''
    Returns even and odd particle number operators in
    Fock space of NN fermions.
    If dense=True keyword argument is given 
    then a dense representation of the operators
    is returned.
    '''
    
    # lists containing the diagonal elements
    even_diag=list(map(lambda x:mod(bin(x).count("1")+1,2),arange(0,2**NN)))
    odd_diag=list(map(lambda x:mod(bin(x).count("1"),2),arange(0,2**NN)))
    # creating a sparse matrix repersenting even and odd projectors
    EVEN=scsp.dia_matrix((even_diag,[0]),shape=(2**NN,2**NN))
    ODD=scsp.dia_matrix((odd_diag,[0]),shape=(2**NN,2**NN))
    
    if dense:
        return [EVEN.todense(),ODD.todense()]
    else:
        return [EVEN,ODD]
    
    
    
def Kitaev_wire_Fock_Ham(cc,t,Delta,mu,**kwargs):
    '''
    Builds Kitaev wire Hamiltonian in Fock space.
    cc    : a list of Fock space annihilation operators 
            assumed to be numpy matrices (not arrays!)
            length of the wire is determined by the number
            of annihilation operators supplied, 
            that is len(cc).
    t     : strength of hopping
    Delta : superconducting pairpotential
    '''
    H_Kit=t*sum( cc[p+1].H*cc[p]+cc[p].H*cc[p+1]   for p in range(len(cc)-1)) \
     +Delta*sum( cc[p+1]*cc[p]+cc[p].H*cc[p+1].H   for p in range(len(cc)-1)) \
        +mu*sum( cc[p].H*cc[p]                     for p in range(len(cc))) 
    return H_Kit

## Useful single particle objects/routines

def Finite_wire_Ham(L,U,T,**kwargs):
    '''
    Returns a Hamiltonian of a finite 1D wire of length L. 
    Hopping is described by T, onsite is described by U.
    
      +-+  T  +-+  T  +-+
    --+U+-----+U+-----+U+-- 
      +-+     +-+     +-+
            
      ⎛..          ⎞
      ⎜  U  T      ⎟ 
    H=⎜  T⁺ U T    ⎟
      ⎜     T⁺ U   ⎟ 
      ⎝         .. ⎠
      
    '''
    idL=eye(L)            # identity matrix of dimension L
    odL=diag(ones(L-1),1) # upper off diagonal matrix with 
                          # ones of size L    
        
    # casting onsite and hopping matrices to matrix type
    # in case they were arrays
    U=matrix(U)          
    T=matrix(T)
    
    return kron(idL,U)+kron(odL,T)+kron(odL,T).H

def Kitaev_wire_BDG_Ham(L,mu,t,Delta):
    '''
    Returns Hamiltonian of a finite length Kitaev wire.
    The degrees of freedom are grouped as
    (c_1,c^dager_1,c_2,c^\dagger_2,...,c_L,c^\dagger_L)
    L     : length of the wire
    t     : strength of hopping
    Delta : superconducting pairpotential
    '''
    U=mu*s3
    T=-t*s3+1.0j*Delta*s2
    return Finite_wire_Ham(L,U,T)
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 10:11:32 2019

@author: Yue Zhang
"""

import numpy as np
from IPython import embed;
import warnings
class qn (np.ndarray):
    qn_dtype = [('w', np.double),('x', np.double),('y', np.double),('z', np.double)]
    def __new__(cls, compactMat):
        mattype = type(compactMat)
        if mattype == list:                  # Check if input is ndarray or list, else return a type error
            compactMat = np.asarray(compactMat)
        elif mattype == np.ndarray:
            pass
        else:
            raise Exception('Input array should be a list or ndarray, instead its type is %s\n' % mattype)
        matshape = compactMat.shape
        qn_compact = np.zeros([*matshape[:-1]],dtype = qn.qn_dtype)
        qn_compact = np.full_like(qn_compact,np.nan)
        if matshape[-1] == 4:
            compactMat_r = compactMat.reshape([-1,4])
            qn_compact['w'] = compactMat_r[:,0].reshape(matshape[:-1])
            qn_compact['x'] = compactMat_r[:,1].reshape(matshape[:-1])
            qn_compact['y'] = compactMat_r[:,2].reshape(matshape[:-1])
            qn_compact['z'] = compactMat_r[:,3].reshape(matshape[:-1])
        elif matshape[-1] == 3:
            targetshape = list(matshape)
            targetshape[-1] = 4
            compactMat_r = compactMat.reshape([-1,3])
            qn_compact['w'] = np.zeros(matshape[:-1])
            qn_compact['x'] = compactMat_r[:,0].reshape(matshape[:-1])
            qn_compact['y'] = compactMat_r[:,1].reshape(matshape[:-1])
            qn_compact['z'] = compactMat_r[:,2].reshape(matshape[:-1])
            warningmsg = "Input array %s is set to %s" %(matshape,tuple(targetshape))
            warnings.warn(warningmsg)
        else:
            raise Exception('Input array should be a N x ... x 4 matrix, instead its shape is %s\n' % (matshape,))
        obj = qn_compact.view(cls)
        # obj.w = qn_compact['w']
        # obj.x = qn_compact['x']
        # obj.y = qn_compact['y']
        # obj.z = qn_compact['z']
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        # self['w'] = getattr(obj, 'w', None)
        # self['x'] = getattr(obj, 'x', None)
        # self['y'] = getattr(obj, 'y', None)
        # self['z'] = getattr(obj, 'z', None)

    def __getitem__(self, keys):
        if type(keys) == int:
            keys = slice(keys,keys+1)
        sub_self = self.view(np.ndarray)[keys]
        # sub_self = super(qn,self).__getitem__(keys)
        if type(keys) != str:
            return sub_self.view(qn)
        else:
            return sub_self.view(np.ndarray)

    def asMatrix(self):
        compact_dtype = np.dtype([('wxyz','f8',4)])
        return self.view(compact_dtype)['wxyz']

    def compact(self):
        return self.asMatrix().reshape(-1,4)

    def __repr__(self):
        concate_qArray = self.compact()
        stringArray = []
        for ci in range(concate_qArray.shape[0]):
            stringArray.append("%+.4g%+.4gi%+.4gj%+.4gk" % tuple(concate_qArray[ci,:]))
        stringOutput = np.array2string(np.asarray(stringArray).reshape(self['w'].shape))
        if len(stringOutput) > 1000//4*4:
            stringOutput = stringOutput[:1000//4*4]+'...'

        return '\n'.join(["Quaternion Array " + str(self.shape)+": ", stringOutput])

    def __neg__(self):
        compactProduct = -self.asMatrix()
        return compactProduct.view(self.qn_dtype).view(qn)

    def __add__(self,qn2):
        if any([1 if (qn2.__class__ == k) else 0 for k in (int,float,np.ndarray,np.float64,np.float32,np.int)]):
            compactProduct = self.asMatrix()+qn2
        elif qn2.__class__ == self.__class__:
            compactProduct = self.asMatrix()+qn2.asMatrix()
        else:
            raise ValueError('Invalid type of input')
        return compactProduct.view(self.qn_dtype).view(qn)

    def __iadd__(self,qn2):
        return self.__add__(qn2)

    def __radd__(self,qn2):
        return self.__add__(qn2)

    def __sub__(self,qn2):
        if any([1 if (qn2.__class__ == k) else 0 for k in (int,float,np.ndarray,np.float64,np.float32,np.int)]):
            compactProduct = self.asMatrix()-qn2
        elif qn2.__class__ == qn:
            compactProduct = self.asMatrix()-qn2.asMatrix()
        else:
            raise ValueError('Invalid type of input')
        return compactProduct.view(self.qn_dtype).view(qn)

    def __isub__(self,qn2):
        return self.__sub__(qn2)

    def __rsub__(self,qn2):
        if any([1 if (qn2.__class__ == k) else 0 for k in (int,float,np.ndarray,np.float64,np.float32,np.int)]):
            compactProduct = qn2-self.asMatrix()
        elif qn2.__class__ == qn:
            compactProduct = qn2.asMatrix()-self.asMatrix()
        else:
            raise ValueError('Invalid type of input')
        return compactProduct.view(self.qn_dtype).view(qn)

    def __mul__(self,qn2):
        if any([1 if (qn2.__class__ == k) else 0 for k in (int,float,np.ndarray,np.float64,np.float32,np.int)]):
            compactProduct = self.asMatrix()*qn2
            compactProduct = compactProduct.view(self.qn_dtype)
        elif qn2.__class__ == self.__class__:
            temp_shape = (self['w']*qn2['w']).shape
            if not temp_shape:
                compactProduct = np.zeros([(self['w']*qn2['w']).size],dtype = self.qn_dtype)
            else:
                compactProduct = np.zeros([*temp_shape],dtype = self.qn_dtype)
            compactProduct = np.full_like(compactProduct,np.nan)
            compactProduct['w'] = self['w']*qn2['w'] - self['x']*qn2['x'] - self['y']*qn2['y'] - self['z']*qn2['z']
            compactProduct['x'] = self['w']*qn2['x'] + self['x']*qn2['w'] + self['y']*qn2['z'] - self['z']*qn2['y']
            compactProduct['y'] = self['w']*qn2['y'] - self['x']*qn2['z'] + self['y']*qn2['w'] + self['z']*qn2['x']
            compactProduct['z'] = self['w']*qn2['z'] + self['x']*qn2['y'] - self['y']*qn2['x'] + self['z']*qn2['w']
        else:
            raise ValueError('Invalid type of input')
        return compactProduct.view(qn)

    def __rmul__(self,qn2):
        if any([1 if (qn2.__class__ == k) else 0 for k in (int,float,np.ndarray,np.float64,np.float32,np.int)]):
            compactProduct = self.asMatrix()*qn2
            compactProduct = compactProduct.view(self.qn_dtype)
        elif qn2.__class__ == self.__class__:
            temp_shape = (self['w']*qn2['w']).shape
            if not temp_shape:
                compactProduct = np.zeros([(self['w']*qn2['w']).size],dtype = self.qn_dtype)
            else:
                compactProduct = np.zeros([*temp_shape],dtype = self.qn_dtype)
            compactProduct = np.full_like(compactProduct,np.nan)
            compactProduct['w'] = qn2['w']*self['w'] - qn2['x']*self['x'] - qn2['y']*self['y'] - qn2['z']*self['z']
            compactProduct['x'] = qn2['w']*self['x'] + qn2['x']*self['w'] + qn2['y']*self['z'] - qn2['z']*self['y']
            compactProduct['y'] = qn2['w']*self['y'] - qn2['x']*self['z'] + qn2['y']*self['w'] + qn2['z']*self['x']
            compactProduct['z'] = qn2['w']*self['z'] + qn2['x']*self['y'] - qn2['y']*self['x'] + qn2['z']*self['w']
        else:
            raise ValueError('Invalid type of input')
        return compactProduct.view(qn)

    def __imul__(self,qn2):
        return self.__mul__(qn2)

    def conj(self):
        conj_num = self.view(np.ndarray)
        conj_num['x'] *= -1
        conj_num['y'] *= -1
        conj_num['z'] *= -1
        return conj_num.view(qn)

    def inv(self):
        qconj = self.conj()
        Q_innerproduct = self*qconj
        Q_ip_inv = 1/Q_innerproduct['w']
        return np.squeeze(qconj*Q_ip_inv[...,None])

    def qT(self):
        return self.conj().T

    def __truediv__(self,qn2):
        if any([1 if (qn2.__class__ == k) else 0 for k in (int,float,np.float64,np.float32,np.int)]):
            compactProduct = self.asMatrix()/qn2
            compactProduct = compactProduct.view(self.qn_dtype)
        elif qn2.__class__ == np.ndarray:
            compactProduct = self.asMatrix()/qn2[...,None]
            compactProduct = compactProduct.view(self.qn_dtype)
        elif qn2.__class__ == self.__class__:
            inv_qn2 = qn2.inv()
            temp_shape = (self['w']*inv_qn2['w']).shape
            if not temp_shape:
                compactProduct = np.zeros([(self['w']*inv_qn2['w']).size],dtype = self.qn_dtype)
            else:
                compactProduct = np.zeros([*temp_shape],dtype = self.qn_dtype)
            compactProduct = np.full_like(compactProduct,np.nan)
            compactProduct['w'] = self['w']*inv_qn2['w'] - self['x']*inv_qn2['x'] - self['y']*inv_qn2['y'] - self['z']*inv_qn2['z']
            compactProduct['x'] = self['w']*inv_qn2['x'] + self['x']*inv_qn2['w'] + self['y']*inv_qn2['z'] - self['z']*inv_qn2['y']
            compactProduct['y'] = self['w']*inv_qn2['y'] - self['x']*inv_qn2['z'] + self['y']*inv_qn2['w'] + self['z']*inv_qn2['x']
            compactProduct['z'] = self['w']*inv_qn2['z'] + self['x']*inv_qn2['y'] - self['y']*inv_qn2['x'] + self['z']*inv_qn2['w']
        else:
            raise ValueError('Invalid type of input')
        return compactProduct.view(qn)

    def __rtruediv__(self,qn2):
        inv_self = self.inv()
        if any([1 if (qn2.__class__ == k) else 0 for k in (int,float,np.float64,np.float32,np.int)]):
            compactProduct = inv_self.asMatrix()/qn2
            compactProduct = compactProduct.view(self.qn_dtype)
        elif qn2.__class__ == np.ndarray:
            compactProduct = inv_self.asMatrix()/qn2[...,None]
            compactProduct = compactProduct.view(self.qn_dtype)
        elif qn2.__class__ == self.__class__:
            temp_shape = (qn2['w']*inv_self['w']).shape
            if not temp_shape:
                compactProduct = np.zeros([(qn2['w']*inv_self['w']).size],dtype = self.qn_dtype)
            else:
                compactProduct = np.zeros([*temp_shape],dtype = self.qn_dtype)
            compactProduct = np.full_like(compactProduct,np.nan)
            compactProduct['w'] = qn2['w']*inv_self['w'] - qn2['x']*inv_self['x'] - qn2['y']*inv_self['y'] - qn2['z']*inv_self['z']
            compactProduct['x'] = qn2['w']*inv_self['x'] + qn2['x']*inv_self['w'] + qn2['y']*inv_self['z'] - qn2['z']*inv_self['y']
            compactProduct['y'] = qn2['w']*inv_self['y'] - qn2['x']*inv_self['z'] + qn2['y']*inv_self['w'] + qn2['z']*inv_self['x']
            compactProduct['z'] = qn2['w']*inv_self['z'] + qn2['x']*inv_self['y'] - qn2['y']*inv_self['x'] + qn2['z']*inv_self['w']
        else:
            raise ValueError('Invalid type of input')
        return compactProduct.view(qn)

    def __itruediv__(self,qn2):
        return self.__truediv__(qn2)

    def sum(self,**kwargs):
        sum_axis = kwargs.pop('axis',None)
        if sum_axis:
            if sum_axis<0:
                sum_axis -= 1
            elif sum_axis>self.ndim:
                raise AxisError('axis %d is out of bounds for array of dimension %d' %(sum_axis,self.ndim))
        else:
            sum_axis = 0
        kwargs['axis'] = sum_axis
        if not self.shape:
            qmatSum = self.asMatrix()
        else:
            qmatSum = np.sum(self.asMatrix(),**kwargs)
        return qmatSum.view(self.qn_dtype).view(qn)

    def imag(self):
        imagpart = np.copy(self)
        imagpart['w'] = 0
        return imagpart.view(qn)

    def real(self):
        realpart = np.copy(self)
        realpart['x'] = 0
        realpart['y'] = 0
        realpart['z'] = 0
        return realpart.view(qn)

    def imagpart(self):
        return self.asMatrix()[...,1:]

    def realpart(self):
        return self['w']

    def norm(self):
        return np.sqrt(np.sum(self.asMatrix()**2,axis= -1))

    def normalize(self):
        return self/self.norm()

def stack(*qn_array,**kwargs):
    stack_axis = kwargs.pop('axis',None)
    if stack_axis == 0:
        qmatStack = np.hstack([x.asMatrix() for x in qn_array],**kwargs)
    elif stack_axis == 1:
        qmatStack = np.vstack([x.asMatrix() for x in qn_array],**kwargs)
    else:
        qmatStack = np.stack([x.asMatrix() for x in qn_array],**kwargs)
    return qmatStack.view(qn.qn_dtype).view(qn)

def sum(*qn_array,**kwargs):
    sum_axis = kwargs.pop('axis',None)
    if sum_axis:
        if sum_axis<0:
            sum_axis -= 1
        elif sum_axis>qn_array[0].ndim:
            raise AxisError('axis %d is out of bounds for array of dimension %d' %(sum_axis,self.ndim))
    else:
        sum_axis = 0
    kwargs['axis'] = sum_axis

    qmatStack = np.squeeze(np.stack([x for x in qn_array],**kwargs))
    if not qmatStack.shape:
        qmatSum = qmatStack.view(qn).asMatrix()
    else:
        qmatSum   = np.sum(qmatStack.view(qn).asMatrix(),**kwargs)
    return qmatSum.view(qn_array[0].qn_dtype).view(qn)

def nansum(*qn_array,**kwargs):
    sum_axis = kwargs.pop('axis',None)
    if sum_axis:
        if sum_axis<0:
            sum_axis -= 1
        elif sum_axis>qn_array[0].ndim:
            raise AxisError('axis %d is out of bounds for array of dimension %d' %(sum_axis,self.ndim))
    else:
        sum_axis = 0
    kwargs['axis'] = sum_axis
    qmatStack = np.squeeze(np.stack([x for x in qn_array],**kwargs))
    if not qmatStack.shape:
        qmatSum = qmatStack.view(qn).asMatrix()
    else:
        qmatSum   = np.nansum(qmatStack.view(qn).asMatrix(),**kwargs)
    return qmatSum.view(qn_array[0].qn_dtype).view(qn)

def mean(*qn_array,**kwargs):
    sum_axis = kwargs.pop('axis',None)
    if sum_axis:
        if sum_axis<0:
            sum_axis -= 1
        elif sum_axis>qn_array[0].ndim:
            raise AxisError('axis %d is out of bounds for array of dimension %d' %(sum_axis,self.ndim))
    else:
        sum_axis = 0
    kwargs['axis'] = sum_axis
    qmatStack = np.squeeze(np.stack([x for x in qn_array],**kwargs))
    if not qmatStack.shape:
        qmatSum = qmatStack.view(qn).asMatrix()
    else:
        qmatSum   = np.mean(qmatStack.view(qn).asMatrix(),**kwargs)
    return qmatSum.view(qn_array[0].qn_dtype).view(qn)

def exp(qn1):
    coeff_real = np.exp(qn1['w'])
    coeff_imag_base = qn1.imag().norm()
    coeff_imag = np.sin(coeff_imag_base)/coeff_imag_base
    temp_shape = qn1['w'].shape
    if not temp_shape:
        compactProduct = np.zeros([qn1['w'].size],dtype = qn1.qn_dtype)
    else:
        compactProduct = np.zeros([*temp_shape],dtype = qn1.qn_dtype)
    compactProduct = np.full_like(compactProduct,np.nan)
    compactProduct['w'] = coeff_real*np.cos(coeff_imag_base)
    compactProduct['x'] = qn1['x']*coeff_imag
    compactProduct['y'] = qn1['y']*coeff_imag
    compactProduct['z'] = qn1['z']*coeff_imag
    return compactProduct.view(qn)


def qdot(qn1,qn2):
    return (qn1*qn2).realpart()

def qcross(qn1,qn2):
    return (qn1*qn2).imag()

def anglebtw(qn1,qn2):
    return np.arccos(qdot(qn1,qn2)/(qn1.norm()*qn2.norm()))

def reflect(surf_normal,points):
    surf_normal /= surf_normal.norm()
    return(surf_normal*points*surf_normal)

def rotate(rot_axis,rot_point,rot_angle = None):
    if type(rot_angle) != type(None):
        rot_axis = np.squeeze(exp(rot_angle/2*rot_axis.normalize()))
    # rot_axis[np.isnan(rot_axis.norm())] *= 0
    return(rot_axis*rot_point*rot_axis.conj())

def rotTo(fromQn,toQn):
    transVec   = fromQn*toQn.normalize()
    transVec   = transVec.imag()-transVec.real()
    transVec   += transVec.norm()
    return transVec.normalize()

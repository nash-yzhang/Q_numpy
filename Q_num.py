# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 10:11:32 2019

@author: Yue Zhang
"""

import numpy as np
import warnings
class qn (np.ndarray):
    qn_dtype = [('w', np.float),('x', np.double),('y', np.double),('z', np.double)]
    def __new__(cls, compactMat):
        mattype = type(compactMat)
        if mattype == list:                  # Check if input is ndarray or list, else return a type error
            compactMat = np.asarray(compactMat)
        elif mattype == np.ndarray:
            pass
        else:
            raise Exception('Input array should be a list or ndarray, instead its type is %s\n' % mattype)
        matshape = compactMat.shape
        if matshape[-1] == 4:
            compactMat_r = compactMat.reshape([-1,4])
            Qn_w = compactMat_r[:,0].reshape(matshape[:-1])
            Qn_x = compactMat_r[:,1].reshape(matshape[:-1])
            Qn_y = compactMat_r[:,2].reshape(matshape[:-1])
            Qn_z = compactMat_r[:,3].reshape(matshape[:-1])
        elif matshape[-1] == 3:
            targetshape = list(matshape)
            targetshape[-1] = 4
            compactMat_r = compactMat.reshape([-1,3])
            Qn_w = np.zeros(matshape[:-1])
            Qn_x = compactMat_r[:,0].reshape(matshape[:-1])
            Qn_y = compactMat_r[:,1].reshape(matshape[:-1])
            Qn_z = compactMat_r[:,2].reshape(matshape[:-1])
            warningmsg = "Input array %s is set to %s" %(matshape,tuple(targetshape))
            warnings.warn(warningmsg)
        else:
            raise Exception('Input array should be a N x ... x 4 matrix, instead its shape is %s\n' % (matshape,))
        qn_compact = np.zeros([*Qn_w.shape],dtype = qn.qn_dtype)
        qn_compact = np.full_like(qn_compact,np.nan)
        qn_compact['w'] = Qn_w
        qn_compact['x'] = Qn_x
        qn_compact['y'] = Qn_y
        qn_compact['z'] = Qn_z
        obj = qn_compact.view(cls)
        obj.w = qn_compact['w']
        obj.x = qn_compact['x']
        obj.y = qn_compact['y']
        obj.z = qn_compact['z']
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.w = getattr(obj, 'w', None)
        self.x = getattr(obj, 'x', None)
        self.y = getattr(obj, 'y', None)
        self.z = getattr(obj, 'z', None)

    def __getitem__(self, keys):
        if type(keys) == int:
            keys = slice(keys,keys+1)
        sub_self = super(qn,self).__getitem__(keys)
        if type(keys) != str:
            sub_self.w = sub_self.w[keys]
            sub_self.x = sub_self.x[keys]
            sub_self.y = sub_self.y[keys]
            sub_self.z = sub_self.z[keys]
            return sub_self
        else:
            return sub_self.view(np.ndarray)

    def asMatrix(self):
        return np.stack([self.w,self.x,self.y,self.z],axis = -1)

    def compact(self):
        return self.asMatrix().reshape(-1,4)

    def __repr__(self):
        concate_qArray = self.compact()
        stringArray = []
        for ci in range(concate_qArray.shape[0]):
            stringArray.append("%+.4g%+.4gi%+.4gj%+.4gk" % tuple(concate_qArray[ci,:]))
        stringOutput = np.array2string(np.asarray(stringArray).reshape(self.w.shape))
        if len(stringOutput) > 1000//4*4:
            stringOutput = stringOutput[:1000//4*4]+'...'

        return '\n'.join(["Quaternion Array " + str(self.shape)+": ", stringOutput])


    def __add__(self,qn2):
        if any([1 if (qn2.__class__ == k) else 0 for k in (int,float,np.ndarray,np.float64,np.float32,np.int)]):
            product_w = self.w+qn2
            product_x = self.x+qn2
            product_y = self.y+qn2
            product_z = self.z+qn2
        elif qn2.__class__ == self.__class__:
            product_w = self.w+qn2.w
            product_x = self.x+qn2.x
            product_y = self.y+qn2.y
            product_z = self.z+qn2.z
        else:
            raise ValueError('Invalid type of input')
        compactProduct = np.stack((product_w,product_x,product_y,product_z),axis = -1)
        return qn(compactProduct)

    def __iadd__(self,qn2):
        return self.__add__(qn2)

    def __radd__(self,qn2):
        return self.__add__(qn2)

    def __sub__(self,qn2):
        if any([1 if (qn2.__class__ == k) else 0 for k in (int,float,np.ndarray,np.float64,np.float32,np.int)]):
            product_w = self.w-qn2
            product_x = self.x-qn2
            product_y = self.y-qn2
            product_z = self.z-qn2
        elif qn2.__class__ == qn:
            product_w = self.w-qn2.w
            product_x = self.x-qn2.x
            product_y = self.y-qn2.y
            product_z = self.z-qn2.z
        else:
            raise ValueError('Invalid type of input')
        compactProduct = np.stack((product_w,product_x,product_y,product_z),axis = -1)
        return qn(compactProduct)

    def __isub__(self,qn2):
        return self.__sub__(qn2)

    def __rsub__(self,qn2):
        return self.__sub__(qn2)

    def __mul__(self,qn2):
        if any([1 if (qn2.__class__ == k) else 0 for k in (int,float,np.ndarray,np.float64,np.float32,np.int)]):
            product_w = self.w*qn2
            product_x = self.x*qn2
            product_y = self.y*qn2
            product_z = self.z*qn2
        elif qn2.__class__ == self.__class__:
            product_w = self.w*qn2.w - self.x*qn2.x - self.y*qn2.y - self.z*qn2.z
            product_x = self.w*qn2.x + self.x*qn2.w + self.y*qn2.z - self.z*qn2.y
            product_y = self.w*qn2.y - self.x*qn2.z + self.y*qn2.w + self.z*qn2.x
            product_z = self.w*qn2.z + self.x*qn2.y - self.y*qn2.x + self.z*qn2.w
        else:
            raise ValueError('Invalid type of input')
        compactProduct = np.stack((product_w,product_x,product_y,product_z),axis = -1)
        return qn(compactProduct)

    def __rmul__(self,qn2):
        if any([1 if (qn2.__class__ == k) else 0 for k in (int,float,np.ndarray,np.float64,np.float32,np.int)]):
            product_w = self.w*qn2
            product_x = self.x*qn2
            product_y = self.y*qn2
            product_z = self.z*qn2
        elif qn2.__class__ == self.__class__:
            product_w = qn2.w*self.w - qn2.x*self.x - qn2.y*self.y - qn2.z*self.z
            product_x = qn2.w*self.x + qn2.x*self.w + qn2.y*self.z - qn2.z*self.y
            product_y = qn2.w*self.y - qn2.x*self.z + qn2.y*self.w + qn2.z*self.x
            product_z = qn2.w*self.z + qn2.x*self.y - qn2.y*self.x + qn2.z*self.w
        else:
            raise ValueError('Invalid type of input')
        compactProduct = np.stack((product_w,product_x,product_y,product_z),axis = -1)
        return qn(compactProduct)

    def __imul__(self,qn2):
        return self.__mul__(qn2)

    def conj(self):
        conj_num = self*-1
        conj_num['w'] = -conj_num['w']
        return conj_num

    def inv(self):
        qconj = self.conj()
        Q_innerproduct = self*qconj
        Q_ip_inv = 1/Q_innerproduct.w
        return qconj*Q_ip_inv

    def qT(self):
        return self.conj().T

    def __truediv__(self,qn2):
        if any([1 if (qn2.__class__ == k) else 0 for k in (int,float,np.ndarray,np.float64,np.float32,np.int)]):
            product_w = self.w/qn2
            product_x = self.x/qn2
            product_y = self.y/qn2
            product_z = self.z/qn2
        elif qn2.__class__ == self.__class__:
            inv_qn2 = qn2.inv()
            product_w = self.w*inv_qn2.w - self.x*inv_qn2.x - self.y*inv_qn2.y - self.z*inv_qn2.z
            product_x = self.w*inv_qn2.x + self.x*inv_qn2.w + self.y*inv_qn2.z - self.z*inv_qn2.y
            product_y = self.w*inv_qn2.y - self.x*inv_qn2.z + self.y*inv_qn2.w + self.z*inv_qn2.x
            product_z = self.w*inv_qn2.z + self.x*inv_qn2.y - self.y*inv_qn2.x + self.z*inv_qn2.w
        else:
            raise ValueError('Invalid type of input')
        compactProduct = np.stack((product_w,product_x,product_y,product_z),axis = -1)
        return qn(compactProduct)

    def __rtruediv__(self,qn2):
        inv_self = self.inv()
        if any([1 if (qn2.__class__ == k) else 0 for k in (int,float,np.ndarray,np.float64,np.float32,np.int)]):
            product_w = inv_self.w*qn2
            product_x = inv_self.x*qn2
            product_y = inv_self.y*qn2
            product_z = inv_self.z*qn2
        elif qn2.__class__ == self.__class__:
            product_w = qn2.w*inv_self.w - qn2.x*inv_self.x - qn2.y*inv_self.y - qn2.z*inv_self.z
            product_x = qn2.w*inv_self.x + qn2.x*inv_self.w + qn2.y*inv_self.z - qn2.z*inv_self.y
            product_y = qn2.w*inv_self.y - qn2.x*inv_self.z + qn2.y*inv_self.w + qn2.z*inv_self.x
            product_z = qn2.w*inv_self.z + qn2.x*inv_self.y - qn2.y*inv_self.x + qn2.z*inv_self.w
        else:
            raise ValueError('Invalid type of input')
        compactProduct = np.stack((product_w,product_x,product_y,product_z),axis = -1)
        return qn(compactProduct)

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
        qmatSum = np.sum(self.asMatrix(),**kwargs)
        return qn(qmatSum)

    def imag(self):
        return (self-self.conj())/2

    def real(self):
        return (self+self.conj())/2

    def imagpart(self):
        return np.stack((self.x,self.y,self.z),axis = -1)

    def realpart(self):
        return self.w

    def norm(self):
        return np.sqrt(np.sum(self.asMatrix()**2,axis= -1))

    def normalize(self):
        return self/self.norm()

def stack(*qn_array,**kwargs):
    qmatStack = np.stack([x.asMatrix() for x in qn_array],**kwargs)
    return qn(qmatStack)

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
    qmatStack = np.squeeze(np.stack([x.asMatrix() for x in qn_array],**kwargs))
    qmatSum   = np.sum(qmatStack,**kwargs)
    return qn(qmatSum)

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
    qmatStack = np.squeeze(np.stack([x.asMatrix() for x in qn_array],**kwargs))
    qmatSum   = np.nansum(qmatStack,**kwargs)
    return qn(qmatSum)

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
    qmatStack = np.squeeze(np.stack([x.asMatrix() for x in qn_array],**kwargs))
    qmatMean   = np.mean(qmatStack,**kwargs)
    return qn(qmatMean)

def exp(qn1):
    coeff_real = np.exp(qn1.w)
    coeff_imag_base = qn1.imag().norm()
    coeff_imag = np.sin(coeff_imag_base)/coeff_imag_base
    qn_exp     =  np.stack([coeff_real*np.cos(coeff_imag_base),
                   qn1.x*coeff_imag,qn1.y*coeff_imag,qn1.z*coeff_imag],axis = -1)
    return qn(qn_exp)


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
        rot_axis = exp(rot_angle/2*rot_axis.normalize())
    rot_axis[np.isnan(rot_axis.norm())] *= 0
    return(rot_axis*rot_point*rot_axis.conj())

def rotTo(fromQn,toQn):
    transVec   = fromQn*toQn.normalize()
    transVec   = transVec.imag()-transVec.real()
    transVec   += transVec.norm()
    return transVec.normalize()

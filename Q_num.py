"""
Created on Fri Aug 9 10:11:32 2019
Last update: Fri Nov 8 17:14:32 2019
@author: Yue Zhang
"""
import numpy as np
import warnings

class qn (np.ndarray):
    """
    Created a quaternion structured ndarray with 4 fields (w,x,y,z) for the 4
    parts of quaternion number (double precision)
    Input:
        - M x ... x N x 4 ndarray. The slices of the last dimension will be assigned to
        4 fields.
        - If the last dimension of the input array only have 3 slices, then the input
        array is assumed to be the coordinates of 3D cartesian space, the slices in
        such an array will be assigned the field x, y, z respectively, the field w
        will be filled with 0. A warning message will be returned to remind the
        assumption of the input data structure.
    Output:
        - M x ... x N quaternion structured array.
    """
    qn_dtype = [('w', np.double),('x', np.double),('y', np.double),('z', np.double)]
    def __new__(cls, compactMat):
        mattype = type(compactMat)
        if mattype == list:                  # Check if input is ndarray or list, else return a type error
            compactMat = np.asarray(compactMat)
        elif mattype == np.ndarray:
            pass
        else:
            raise Exception('Input array should be a list or ndarray, instead its type is %s\n' % mattype)
        matshape = compactMat.shape          # Shape checking: should be M x ... x 4 or x 3 ndarray, otherwise return error
        qn_compact = np.zeros([*matshape[:-1]],dtype = qn.qn_dtype) # Preallocate space
        qn_compact = np.full_like(qn_compact,np.nan)                # filled with nan for debugging.
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
        obj = qn_compact.view(cls)                                  # Convert to quaternion ndarray object
        return obj

    ###################################  Method  ###################################

    def __getitem__(self, keys):
        """
        Custom indexing for structured quaternion ndarray
        """
        if type(keys) == int: # If index is integer, converted to slices; otherwise cause error
            keys = slice(keys,keys+1)
        sub_self = self.view(np.ndarray)[keys]
        # sub_self = super(qn,self).__getitem__(keys)
        if type(keys) != str:
            return sub_self.view(qn)
        else:
            return sub_self.view(np.ndarray)

    def asMatrix(self):
        # Converted to the double M x ... x 4 unstructured ndarray
        compact_dtype = np.dtype([('wxyz','double',4)])
        return self.view(compact_dtype)['wxyz']

    def compact(self):
        # Converted to double (Mx...xN) x 4 unstructured ndarray
        return self.asMatrix().reshape(-1,4)

    def __repr__(self):
        """
        Custom representation for the quaternion array. each quaternion number will be
        show as "a+bi+cj+dk". The printed string are organized in the same way
        as the quaternion ndarray
        """
        concate_qArray = self.compact()
        stringArray = []
        for ci in range(concate_qArray.shape[0]):
            stringArray.append("%+.4g%+.4gi%+.4gj%+.4gk" % tuple(concate_qArray[ci,:]))
        stringOutput = np.array2string(np.asarray(stringArray).reshape(self['w'].shape))
        if len(stringOutput) > 1000//4*4:
            stringOutput = stringOutput[:1000//4*4]+'...'

        return '\n'.join(["Quaternion Array " + str(self.shape)+": ", stringOutput])

    def __neg__(self):
        # Elementary arithmetic: qn * -1
        compactProduct = -self.asMatrix()
        return np.reshape(compactProduct.view(self.qn_dtype).view(qn),compactProduct.shape[:-1])

    def __add__(self,qn2):
        # Elementary arithmetic: qn1 + qn2 or qn1 + r (real number). Same as the elementary arithmetic for real number
        if any([1 if (qn2.__class__ == k) else 0 for k in (int,float,np.ndarray,np.float64,np.float32,np.int)]):
            compactProduct = self.asMatrix()+qn2
        elif qn2.__class__ == self.__class__:
            compactProduct = self.asMatrix()+qn2.asMatrix()
        else:
            raise ValueError('Invalid type of input')
        return np.reshape(compactProduct.view(self.qn_dtype).view(qn),compactProduct.shape[:-1])

    def __iadd__(self,qn2):
        # Elementary arithmetic: qn1 += qn2 or qn1 += r
        return self.__add__(qn2)

    def __radd__(self,qn2):
        # Elementary arithmetic: qn2 + qn1 or r + qn1
        return self.__add__(qn2)

    def __sub__(self,qn2):
        # Elementary arithmetic: qn1 - qn2. Same as the elementary arithmetic for real number
        if any([1 if (qn2.__class__ == k) else 0 for k in (int,float,np.ndarray,np.float64,np.float32,np.int)]):
            compactProduct = self.asMatrix()-qn2
        elif qn2.__class__ == qn:
            compactProduct = self.asMatrix()-qn2.asMatrix()
        else:
            raise ValueError('Invalid type of input')
        return np.reshape(compactProduct.view(self.qn_dtype).view(qn),compactProduct.shape[:-1])

    def __isub__(self,qn2):
        # Elementary arithmetic: qn1 -= qn2 or qn1 -= r
        return self.__sub__(qn2)

    def __rsub__(self,qn2):
        # Elementary arithmetic: qn2 - qn1 or r - qn1
        if any([1 if (qn2.__class__ == k) else 0 for k in (int,float,np.ndarray,np.float64,np.float32,np.int)]):
            compactProduct = qn2-self.asMatrix()
        elif qn2.__class__ == qn:
            compactProduct = qn2.asMatrix()-self.asMatrix()
        else:
            raise ValueError('Invalid type of input')
        return np.reshape(compactProduct.view(self.qn_dtype).view(qn),compactProduct.shape[:-1])

    def __mul__(self,qn2):
        # Elementary arithmetic: qn1 * qn2; check https://en.wikipedia.org/wiki/Quaternion#Algebraic_properties for details
        if any([1 if (qn2.__class__ == k) else 0 for k in (int,float,np.ndarray,np.float64,np.float32,np.int)]):
            # if qn1 * r, then the same as real number calculation
            compactProduct = self.asMatrix()*qn2
            compactProduct = np.reshape(compactProduct.view(self.qn_dtype),compactProduct.shape[:-1])
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
        # Elementary arithmetic: qn2 * qn1; Note the result of _rmul_ and _mul_ are not equal for quaternion
        if any([1 if (qn2.__class__ == k) else 0 for k in (int,float,np.ndarray,np.float64,np.float32,np.int)]):
            compactProduct = self.asMatrix()*qn2
            compactProduct = np.reshape(compactProduct.view(self.qn_dtype),compactProduct.shape[:-1])
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
        # Elementary arithmetic: qn1 *= qn2; check https://en.wikipedia.org/wiki/Quaternion#Algebraic_properties for details
        return self.__mul__(qn2)

    def conj(self):
        # Conjugate: conj(a+bi+cj+dk) = a-bi-cj-dk
        conj_num = self.view(np.ndarray)
        conj_num['x'] *= -1
        conj_num['y'] *= -1
        conj_num['z'] *= -1
        return conj_num.view(qn)

    def inv(self):
        # 1/qn
        qconj = self.conj()
        Q_innerproduct = self*qconj
        Q_ip_inv = 1/Q_innerproduct['w']
        # The broadcast calculation is necessary here, but need to take care of the redundant dimension otherwise will run into dimensionality expansion problem all the time
        return np.squeeze(qconj*Q_ip_inv[...,None])

    def qT(self):
        # Transposition of quaternion array returns the conjugated quaternions
        # If want transposition without getting the conjugate number, use .T
        return self.conj().T

    def __truediv__(self,qn2):
        # Elementary arithmetic: qn1 / qn2; Note the result of __truediv__ and __rtruediv__ are not equal for quaternion
        if any([1 if (qn2.__class__ == k) else 0 for k in (int,float,np.float64,np.float32,np.int)]):
            compactProduct = self.asMatrix()/qn2
            compactProduct = np.reshape(compactProduct.view(self.qn_dtype),compactProduct.shape[:-1])
        elif qn2.__class__ == np.ndarray:
            compactProduct = self.asMatrix()/qn2[...,None]
            compactProduct = np.reshape(compactProduct.view(self.qn_dtype),compactProduct.shape[:-1])
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
        # Elementary arithmetic: qn2 / qn1; Note the result of __truediv__ and __rtruediv__ are not equal for quaternion
        inv_self = self.inv()
        if any([1 if (qn2.__class__ == k) else 0 for k in (int,float,np.float64,np.float32,np.int)]):
            compactProduct = inv_self.asMatrix()/qn2
            compactProduct = np.reshape(compactProduct.view(self.qn_dtype),compactProduct.shape[:-1])
        elif qn2.__class__ == np.ndarray:
            compactProduct = inv_self.asMatrix()/qn2[...,None]
            compactProduct = np.reshape(compactProduct.view(self.qn_dtype),compactProduct.shape[:-1])
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
        # Elementary arithmetic: qn1 /= qn2 (or r)
        return self.__truediv__(qn2)

    def sum(self,**kwargs):
        # Not recommended, use the Q_num.sum function instead
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
        # Return the quatenrion number whose real part (qn['w']) = 0;
        imagpart = np.copy(self)
        imagpart['w'] = 0
        return imagpart.view(qn)

    def real(self):
        # Return the quatenrion number whose imag part (qn['xyz']) = 0;
        realpart = np.copy(self)
        realpart['x'] = 0
        realpart['y'] = 0
        realpart['z'] = 0
        return realpart.view(qn)

    def imagpart(self):
        # Return the double real number matrix (M x ... x 3) of the imaginary part
        return self.asMatrix()[...,1:]

    def realpart(self):
        # Return the double real number matrix (M x ... x1) of the real part
        return self['w']

    def norm(self):
        # Return the norm (or the absolute value) of the quaternion number
        return np.sqrt(np.sum(self.asMatrix()**2,axis= -1))

    def normalize(self):
        # Return the normalized quaternion number (norm = 1)
        return self/self.norm()

################################### Functions ###################################

def stack(*qn_array,**kwargs):
    # Same as np.stack
    stack_axis = kwargs.pop('axis',None)
    if stack_axis == 0:
        qmatStack = np.hstack([x.asMatrix() for x in qn_array],**kwargs)
    elif stack_axis == 1:
        qmatStack = np.vstack([x.asMatrix() for x in qn_array],**kwargs)
    else:
        qmatStack = np.stack([x.asMatrix() for x in qn_array],**kwargs)
    return qmatStack.view(qn.qn_dtype).view(qn)

def sum(*qn_array,**kwargs):
    # Same as np.sum
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
    # Same as np.nansum, calculate the sum but ignore nan numbers
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
    # Same as np.mean
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
    # Exponetial calculation for quaternion numbers. Note the qn**2 is still not implemented
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
    # Return the dot product of two quaternion number (as real number ndarray object)
    return -(qn1*qn2).realpart()

def qcross(qn1,qn2):
    # Return the cross product of two quaternion number (as quaternion ndarray object)
    return (qn1*qn2).imag()

def anglebtw(qn1,qn2):
    # Calculate the angle between 3d vectors represented with two quaternions whose real part = 0
    return np.arcsin((qn1.normalize()-qn2.normalize()).norm()/2)*2
    # return np.arccos(qdot(qn1,qn2)/(qn1.norm()*qn2.norm())) # deprecated, too slow


def reflect(surf_normal,points):
    """
    Calculate the reflected 3d vectors representing with quaternions whose real part = 0
    Input:
        surf_normal: normal vector for the reflection surface (quaternion)
        points: qn vectors or points to be reflected
    Output:
        reflected qn vectors/points
    """
    #
    surf_normal /= surf_normal.norm()
    return(surf_normal*points*surf_normal)

def ortho_project(surf_normal,points,on_plane = True):
    """
    Calculate the orthognal projection of quaternion 3d vectors
    Input:
        surf_normal: normal vector for the projection surface (quaternion)
        points: qn vectors or points to be projected
    Output:
        projected qn vectors/points
    """
    if on_plane:
        return (points+reflect(surf_normal,points))/2
    else:
        return (points-reflect(surf_normal,points))/2

def rotate(rot_axis,rot_point,rot_angle = None):
    """
    Perform 3D rotation
    Input:
        rot_axis: rotation axis (in quatennion ndarray form).
        rot_point: qn vectors or points to be rotated
        rot_angle (optional): if exist, it will update the qn number of the rot_axis
         with its value, if not applied, the rotation will be calculated only based on
         the rotation axis qn number
    Output:
        rotated qn vector/points
    """
    if type(rot_angle) != type(None):
        rot_axis = np.squeeze(exp(rot_angle/2*rot_axis.normalize()))
    # rot_axis[np.isnan(rot_axis.norm())] *= 0
    return(rot_axis*rot_point*rot_axis.conj())

def rotTo(fromQn,toQn):
    """
    Given the qn representing the current 3D orientation represented and the qn
    for the target 3D orientation, compute the rotation vectors to transform the
    current orienttaion to the target orientation
    Input:
        fromQn: current orientation qn vectors
        toQn:   target orientation qn vectors
    Output:
        the rotation quaternion number for the rotation transform
    """
    transVec   = fromQn*toQn.normalize()
    transVec   = transVec.imag()-transVec.real()
    transVec   += transVec.norm()
    return transVec.normalize()

/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

		http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
/* Modified by martin.croome@greenwaves-technologies.com - Modifications to allow a standalone build
   and remove requirements for pybind11 and other tensorflow dependencies
   Add support for scalar operations and python numeric types
*/

#include <iostream>
#include <array>
#include <locale>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
// #define DEBUG_CALLS

#include <Python.h>
#include <cinttypes>
#include <vector>
#ifdef DEBUG_CALLS
#include <iostream>
#endif
#include "../include/eigen/Eigen/Core"
#include "../include/posit8/include/universal/number/posit/posit.hpp"
#include "../include/posit8/include/universal/number/posit/posit_c_api.h"
#include "../include/posit8/include/universal/number/posit/positctypes.h"
#include "../include/posit8/include/universal/adapters/adapt_integer_and_posit.hpp"
#include "../include/posit8/include/universal/number/posit/posit_impl.hpp"
#include <fenv.h>
#include "numpy/arrayobject.h"
#include "numpy/ufuncobject.h"
#include <typeinfo>

namespace xposit8
{
	namespace
	{

		using posit8_2 = sw::universal::posit<8UL, 2UL>;
        using uint8 = std::uint8_t;
		using int8 = std::int8_t;
		using uint16 = std::uint16_t;
		using int16 = std::int16_t;
		using uint64 = std::uint64_t;

		// Representation of a Python posit8_2 object.
		struct PyPosit8_2
		{
			PyObject_HEAD; // Python object header
			posit8_2 value;
		};

		void unmarshallPositRaw(PyObject *data, posit8_t &out) {
			posit8_2 num = reinterpret_cast<PyPosit8_2 *>(data)->value;
			int nrBytes = 0;
			int maxBitsInByte = 8;

			sw::universal::bitblock<8UL> raw = num.get();
			nrBytes = 1;

			uint32_t bit_cntr = 0;
			for (int c = 0; c < nrBytes; ++c) {
				unsigned char byte = 0;
				unsigned char mask = (unsigned char) (1);
				for (int b = 0; b < maxBitsInByte; ++b) {
					if (raw[bit_cntr++]) {
						byte |= mask;
					}
					mask <<= 1;
				}
				out.x[c] = byte;
			}
		}
		
		// marshal takes a posit8_t and marshals it into a raw bitblock
		template<size_t nbits, size_t es, typename posit8_t>
		void marshal(posit8_t a, sw::universal::bitblock<8UL>& raw) {
			int nrBytes = 0;
			int maxBitsInByte = 8; // default is multi-byte data structures
			switch (nbits) {
			case 4:
				maxBitsInByte = 4; // except for nbits < 8
			case 8:
				nrBytes = 1;
				break;
			case 16:
				nrBytes = 2;
				break;
			case 32:
				nrBytes = 4;
				break;
			case 64:
				nrBytes = 8;
				break;
			case 128:
				nrBytes = 16;
				break;
			case 256:
				nrBytes = 32;
				break;
			default:
				nrBytes = 0;
			}
			uint32_t bit_cntr = 0;
			for (int c = 0; c < nrBytes; ++c) {
				unsigned char byte = a.x[c];
				unsigned char mask = (unsigned char)(1);
				for (int b = 0; b < maxBitsInByte; ++b) {
					raw[bit_cntr++] = mask & byte;
					mask <<= 1;
				}
			}
		}

		// unmarshal takes a raw bitblock and unmarshals it into a posit8_t
		template<size_t nbits, size_t es, typename posit8_t>
		void unmarshal(sw::universal::bitblock<8UL>& raw, posit8_t& a) {
			int nrBytes = 0;
			int maxBitsInByte = 8; // default is multi-byte data structures
			switch (nbits) {
			case 4:
				maxBitsInByte = 4; // except for nbits < 8
			case 8:
				nrBytes = 1;
				break;
			case 16:
				nrBytes = 2;
				break;
			case 32:
				nrBytes = 4;
				break;
			case 64:
				nrBytes = 8;
				break;
			case 128:
				nrBytes = 16;
				break;
			case 256:
				nrBytes = 32;
				break;
			default:
				nrBytes = 0;
			}
			uint32_t bit_cntr = 0;
			for (int c = 0; c < nrBytes; ++c) {
				unsigned char byte = 0;
				unsigned char mask = (unsigned char)(1);
				for (int b = 0; b < maxBitsInByte; ++b) {
					if (raw[bit_cntr++]) {
						byte |= mask;
					}
					mask <<= 1;
				}
				a.x[c] = byte;
			}
		}

		// from posit raw convert to posit<_, _>
		sw::universal::posit<8UL, 2UL> decode(posit8_t bits) {
			sw::universal::posit<8UL, 2UL> pa;
			sw::universal::bitblock<8UL> raw;
			marshal<8UL, 2UL>(bits, raw);
			pa.setBitblock(raw);
			return pa;
		}

		// from posit<_, _> convert to posit raw
		posit8_t encode(sw::universal::posit<8UL, 2UL> p) {
			posit8_t out;
			sw::universal::bitblock<8UL> raw = p.get();
			unmarshal<8UL, 2UL>(raw, out);
			return out;
		}

		template<class out>
		static out to(posit8_t bits) {
			using namespace sw::universal;
			posit<8UL, 2UL> pa = decode(bits);
			return static_cast<out>(pa);
		}

		template<class in>
		static posit8_t from(in a) {
			using namespace sw::universal;
			posit<8UL, 2UL> pa(a);
			return encode(pa);
		}

		struct PyDecrefDeleter
		{
			void operator()(PyObject *p) const { Py_DECREF(p); }
		};

		// Safe container for an owned PyObject. On destruction, the reference count of
		// the contained object will be decremented.
		using Safe_PyObjectPtr = std::unique_ptr<PyObject, PyDecrefDeleter>;
		Safe_PyObjectPtr make_safe(PyObject *object)
		{
			return Safe_PyObjectPtr(object);
		}

		bool PyLong_CheckNoOverflow(PyObject *object)
		{
			if (!PyLong_Check(object))
			{
				return false;
			}
			int overflow = 0;
			PyLong_AsLongAndOverflow(object, &overflow);
			return (overflow == 0);
		}

		// Registered numpy type ID. Global variable populated by the registration code.
		// Protected by the GIL.
		int npy_posit8_2 = NPY_NOTYPE;

		// Forward declaration.
		extern PyTypeObject posit8_2_type;
		extern PyArray_Descr NPyPosit8_2_Descr;

		// Pointer to the posit8_2 type object we are using. This is either a pointer
		// to Posit8_2_type, if we choose to register it, or to the posit8_2 type
		// registered by another system into NumPy.
		PyTypeObject *posit8_2_type_ptr = nullptr;

		// Returns true if 'object' is a PyPosit8_2.
		bool PyPosit8_2_Check(PyObject *object)
		{
			return PyObject_IsInstance(object, reinterpret_cast<PyObject *>(&posit8_2_type));
		}

		// Extracts the value of a PyPosit8_2 object.
		posit8_2 PyPosit8_2_Posit8_2(PyObject *object)
		{
			return reinterpret_cast<PyPosit8_2 *>(object)->value;
		}

		// Constructs a PyPosit8_2 object from a posit8_2.
		PyObject *PyPosit8_2_FromPosit8_2(posit8_2 x)
		{
			return PyArray_Scalar(&x, &NPyPosit8_2_Descr, NULL);
		}

		// Converts a Python object to a posit8_2 value. Returns true on success,
		// returns false and reports a Python error on failure.
		bool CastToPosit8_2(PyObject *arg, posit8_2 *output)
		{
			if (PyPosit8_2_Check(arg))
			{
				*output = PyPosit8_2_Posit8_2(arg);
				return true;
			}
			if (PyFloat_Check(arg))
			{
				double d = PyFloat_AsDouble(arg);
				if (PyErr_Occurred())
				{
					return false;
				}
				// TODO(phawkins): check for overflow
				*output = posit8_2(d);
				return true;
			}
			if (PyLong_CheckNoOverflow(arg))
			{
				long l = PyLong_AsLong(arg); // NOLINT
				if (PyErr_Occurred())
				{
					return false;
				}
				// TODO(phawkins): check for overflow
				*output = posit8_2(static_cast<float>(l));
				return true;
			}
			//if (PyArray_IsScalar(arg, Half))
			//{
			//	Eigen::half f;
			//	PyArray_ScalarAsCtype(arg, &f);
			//	*output = posit8_2(f);
			//	return true;
			//}
			if (PyArray_IsScalar(arg, Float))
			{
				float f;
				PyArray_ScalarAsCtype(arg, &f);
				*output = posit8_2(f);
				return true;
			}
			if (PyArray_IsScalar(arg, Double))
			{
				double f;
				PyArray_ScalarAsCtype(arg, &f);
				*output = posit8_2(f);
				return true;
			}
			if (PyArray_IsZeroDim(arg))
			{
				Safe_PyObjectPtr ref;
				PyArrayObject *arr = reinterpret_cast<PyArrayObject *>(arg);
				if (PyArray_TYPE(arr) != npy_posit8_2)
				{
					ref = make_safe(PyArray_Cast(arr, npy_posit8_2));
					if (PyErr_Occurred())
					{
						return false;
					}
					arg = ref.get();
					arr = reinterpret_cast<PyArrayObject *>(arg);
				}
				*output = *reinterpret_cast<posit8_2 *>(PyArray_DATA(arr));
				return true;
			}
			return false;
		}

		// Constructs a new PyPosit8_2.
		PyObject *PyPosit8_2_New(PyTypeObject *type, PyObject *args, PyObject *kwds)
		{
			if (kwds && PyDict_Size(kwds))
			{
				PyErr_SetString(PyExc_TypeError, "constructor takes no keyword arguments");
				return nullptr;
			}
			Py_ssize_t size = PyTuple_Size(args);
			if (size != 1)
			{
				PyErr_SetString(PyExc_TypeError,
								"expected number as argument to posit8_2 constructor");
				return nullptr;
			}
			PyObject *arg = PyTuple_GetItem(args, 0);

			posit8_2 value;
			if (PyPosit8_2_Check(arg))
			{
				Py_INCREF(arg);
				return arg;
			}
			else if (CastToPosit8_2(arg, &value))
			{
				return PyPosit8_2_FromPosit8_2(value);
			}
			else if (PyArray_Check(arg))
			{
				PyArrayObject *arr = reinterpret_cast<PyArrayObject *>(arg);
				if (PyArray_TYPE(arr) != npy_posit8_2)
				{
					return PyArray_Cast(arr, npy_posit8_2);
				}
				else
				{
					Py_INCREF(arg);
					return arg;
				}
			}
			PyErr_Format(PyExc_TypeError, "expected number, got %s",
						 arg->ob_type->tp_name);
			return nullptr;
		}

		// Comparisons on PyPosit8_2s.
		PyObject *PyPosit8_2_RichCompare(PyObject *self, PyObject *other, int cmp_op)
		{
			PyObject *arr, *ret;

			arr = PyArray_FromScalar(self, NULL);
			if (arr == NULL)
			{
				return NULL;
			}
			if (PyPosit8_2_Check(other))
			{
				PyObject *arr_other;
				arr_other = PyArray_FromScalar(other, NULL);
				ret = Py_TYPE(arr)->tp_richcompare(arr, arr_other, cmp_op);
				Py_DECREF(arr_other);
			} else {
				ret = Py_TYPE(arr)->tp_richcompare(arr, other, cmp_op);
			}
			Py_DECREF(arr);
			return ret;
		}

		// Implementation of repr() for PyPosit8_2.
		PyObject *PyPosit8_2_Repr(PyObject *self)
		{
			posit8_2 x = reinterpret_cast<PyPosit8_2 *>(self)->value;
			std::string v = std::to_string(static_cast<float>(x));
			return PyUnicode_FromString(v.c_str());
		}

		// Implementation of str() for PyPosit8_2.
		PyObject *PyPosit8_2_Str(PyObject *self)
		{
			posit8_2 x = reinterpret_cast<PyPosit8_2 *>(self)->value;
			std::string v = std::to_string(static_cast<float>(x));
			return PyUnicode_FromString(v.c_str());
		}

		// Hash function for PyPosit8_2. We use the identity function, which is a weak
		// hash function.
		Py_hash_t PyPosit8_2_Hash(PyObject *self)
		{
			posit8_t out;
			unmarshallPositRaw(self, out);
			//unmarshal<8UL, 2UL>(raw, out);
			return out.v;
		}

		// Converts a PyPosit8_2 into a PyFloat.
		PyObject* PyPosit8_2_Float(PyObject* self) {
			posit8_2 x = PyPosit8_2_Posit8_2(self);
			return PyFloat_FromDouble(static_cast<double>(x));
		}

		// Converts a PyPosit8_2 into a PyInt.
		PyObject* PyPosit8_2_Int(PyObject* self) {
			posit8_2 x = PyPosit8_2_Posit8_2(self);
			long y = static_cast<long>(x);  // NOLINT
			return PyLong_FromLong(y);
		}

		PyNumberMethods PyPosit8_2_AsNumber = {
			nullptr,     	  	// nb_add
			nullptr,  			// nb_subtract
			nullptr,  			// nb_multiply
			nullptr,              // nb_remainder
			nullptr,              // nb_divmod
			nullptr,              // nb_power
			nullptr,  			  // nb_negative
			nullptr,              // nb_positive
			nullptr,              // nb_absolute
			nullptr,              // nb_nonzero
			nullptr,              // nb_invert
			nullptr,              // nb_lshift
			nullptr,              // nb_rshift
			nullptr,              // nb_and
			nullptr,              // nb_xor
			nullptr,              // nb_or
			PyPosit8_2_Int,       // nb_int
			nullptr,              // reserved
			PyPosit8_2_Float,     // nb_float

			nullptr,  // nb_inplace_add
			nullptr,  // nb_inplace_subtract
			nullptr,  // nb_inplace_multiply
			nullptr,  // nb_inplace_remainder
			nullptr,  // nb_inplace_power
			nullptr,  // nb_inplace_lshift
			nullptr,  // nb_inplace_rshift
			nullptr,  // nb_inplace_and
			nullptr,  // nb_inplace_xor
			nullptr,  // nb_inplace_or

			nullptr,                // nb_floor_divide
			nullptr,  				// nb_true_divide
			nullptr,                // nb_inplace_floor_divide
			nullptr,                // nb_inplace_true_divide
			nullptr,                // nb_index
		};

		// format posit8_2. Convert to a float and call format on that
		PyObject *PyPosit8_2_Format(PyObject *self, PyObject *format)
		{
			posit8_2 x = reinterpret_cast<PyPosit8_2 *>(self)->value;
			PyObject * f_obj = PyFloat_FromDouble(static_cast<double>(x));
			PyObject * __format__str = PyUnicode_FromString("__format__");
			PyObject * f_str = PyObject_CallMethodObjArgs(f_obj, __format__str, format, NULL);
			Py_DECREF(__format__str);
			Py_XDECREF(f_obj);
			return f_str;
		}

		static PyMethodDef PyPosit8_2_methods[] = {
			{
				"__format__",
				(PyCFunction) PyPosit8_2_Format,
				METH_O,
				"__format__ method for posit8_2"
			},
			{NULL}  /* Sentinel */
		};


//#ifdef IMPLEMENT_BUFFER
//		int PyPosit8_2_getbuffer(PyObject *exporter, Py_buffer *view, int flags) {
//			view->obj = exporter;
//			Py_INCREF(exporter);
//			view->buf = &(reinterpret_cast<PyPosit8_2 *>(exporter)->value);
//			view->len = 1;
//			view->itemsize = sizeof(posit8_2);
//			view->readonly = 0;
//			view->format = NULL;
//			if ((flags & PyBUF_FORMAT) == PyBUF_FORMAT)
//				view->format = (char *)"BB";
//			view->ndim = 1;
//			view->shape = NULL;
//			if ((flags & PyBUF_ND) == PyBUF_ND)
//				view->shape = &(view->len);
//			view->strides = NULL;
//			if ((flags & PyBUF_STRIDES) == PyBUF_STRIDES)
//				view->strides = &(view->itemsize);
//			view->suboffsets = NULL;
//			view->internal = NULL;
//			return 0;
//		}
//
//		static PyBufferProcs PyPosit8_2_buffer_procs = {
//			&PyPosit8_2_getbuffer,
//			NULL
//		};
//#endif

		// Python type for PyPosit8_2 objects.

		PyTypeObject posit8_2_type = {
			PyVarObject_HEAD_INIT(nullptr, 0) "posit8_2", // tp_name
			sizeof(PyPosit8_2),							  // tp_basicsize
			0,											  // tp_itemsize
			nullptr,									  // tp_dealloc
#if PY_VERSION_HEX < 0x03080000
			nullptr, // tp_print
#else
			0, // tp_vectorcall_offset
#endif
			nullptr,			  // tp_getattr
			nullptr,			  // tp_setattr
			nullptr,			  // tp_compare / tp_reserved
			PyPosit8_2_Repr,	  // tp_repr
			&PyPosit8_2_AsNumber, // tp_as_number
			nullptr,			  // tp_as_sequence
			nullptr,			  // tp_as_mapping
			PyPosit8_2_Hash,	  // tp_hash
			nullptr,			  // tp_call
			PyPosit8_2_Str,		  // tp_str
			nullptr,			  // tp_getattro
			nullptr,			  // tp_setattro
#ifdef IMPLEMENT_BUFFER
			&PyPosit8_2_buffer_procs,			  // tp_as_buffer
#else
			nullptr,
#endif
								  // tp_flags
			Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
			"posit8_2 floating-point values", // tp_doc
			nullptr,						  // tp_traverse
			nullptr,						  // tp_clear
			PyPosit8_2_RichCompare,			  // tp_richcompare
			0,								  // tp_weaklistoffset
			nullptr,						  // tp_iter
			nullptr,						  // tp_iternext
			PyPosit8_2_methods,						  // tp_methods
			nullptr,						  // tp_members
			nullptr,						  // tp_getset
			nullptr,			  // tp_base
			nullptr,						  // tp_dict
			nullptr,						  // tp_descr_get
			nullptr,						  // tp_descr_set
			0,								  // tp_dictoffset
			nullptr,						  // tp_init
			nullptr,						  // tp_alloc
			PyPosit8_2_New,					  // tp_new
			nullptr,						  // tp_free
			nullptr,						  // tp_is_gc
			nullptr,						  // tp_bases
			nullptr,						  // tp_mro
			nullptr,						  // tp_cache
			nullptr,						  // tp_subclasses
			nullptr,						  // tp_weaklist
			nullptr,						  // tp_del
			0,								  // tp_version_tag
		};


		// Numpy support

		PyArray_ArrFuncs NPyPosit8_2_ArrFuncs;

		PyArray_Descr NPyPosit8_2_Descr = {
			PyObject_HEAD_INIT(nullptr) //
										/*typeobj=*/
			(&posit8_2_type),
			// We must register posit8_2 with a kind other than "f", because numpy
			// considers two types with the same kind and size to be equal, but
			// float16 != posit8_2.
			// The downside of this is that NumPy scalar promotion does not work with
			// posit8_2 values.
			/*kind=*/'p',
			// TODO(phawkins): there doesn't seem to be a way of guaranteeing a type
			// character is unique.
			/*type=*/'E',
			/*byteorder=*/'=',
			/*flags=*/NPY_NEEDS_PYAPI, // | NPY_USE_GETITEM | NPY_USE_SETITEM,
			/*type_num=*/0,
			/*elsize=*/sizeof(posit8_2),
			/*alignment=*/alignof(posit8_2),
			/*subarray=*/nullptr,
			/*fields=*/nullptr,
			/*names=*/nullptr,
			/*f=*/&NPyPosit8_2_ArrFuncs,
			/*metadata=*/nullptr,
			/*c_metadata=*/nullptr,
			/*hash=*/-1, // -1 means "not computed yet".
		};

		// Implementations of NumPy array methods.

		PyObject *NPyPosit8_2_GetItem(void *data, void  *arr)
		{

			posit8_2 x;
			NPyPosit8_2_Descr.f->copyswap(&x, data, !PyArray_ISNOTSWAPPED(reinterpret_cast<PyArrayObject *>(arr)), NULL);
			return PyPosit8_2_FromPosit8_2(x);
		}

		int NPyPosit8_2_SetItem(PyObject *item, void *data, void *arr)
		{
			posit8_2 x;
			if (!CastToPosit8_2(item, &x))
			{
				PyErr_Format(PyExc_TypeError, "expected number, got %s",
							 item->ob_type->tp_name);
				return -1;
			}
			memcpy(data, &x, sizeof(posit8_2));
			return 0;
		}

		void ByteSwap16(void *value)
		{
			char *p = reinterpret_cast<char *>(value);
			std::swap(p[0], p[1]);
		}

		void NPyPosit8_2_CopySwapN(void *dstv, npy_intp dstride, void *srcv,
								   npy_intp sstride, npy_intp n, int swap, void *arr)
		{
			char *dst = reinterpret_cast<char *>(dstv);
			char *src = reinterpret_cast<char *>(srcv);
			if (!src)
			{
				return;
			}
			if (swap)
			{
				for (npy_intp i = 0; i < n; i++)
				{
					char *r = dst + dstride * i;
					memcpy(r, src + sstride * i, sizeof(uint16_t));
					ByteSwap16(r);
				}
			}
			else if (dstride == sizeof(uint16_t) && sstride == sizeof(uint16_t))
			{
				memcpy(dst, src, n * sizeof(uint16_t));
			}
			else
			{
				for (npy_intp i = 0; i < n; i++)
				{
					memcpy(dst + dstride * i, src + sstride * i, sizeof(uint16_t));
				}
			}
		}

		void NPyPosit8_2_CopySwap(void *dst, void *src, int swap, void *arr)
		{
			if (!src)
			{
				return;
			}
			memcpy(dst, src, sizeof(uint16_t));
			if (swap)
			{
				ByteSwap16(dst);
			}
		}

		npy_bool NPyPosit8_2_NonZero(void *data, void *arr)
		{
			posit8_2 x;
			memcpy(&x, data, sizeof(x));
			return x != static_cast<posit8_2>(0);
		}

		int NPyPosit8_2_Fill(void *buffer_raw, npy_intp length, void *ignored)
		{
			posit8_2 *const buffer = reinterpret_cast<posit8_2 *>(buffer_raw);
			const float start(buffer[0]);
			const float delta = static_cast<float>(buffer[1]) - start;
			for (npy_intp i = 2; i < length; ++i)
			{
				buffer[i] = static_cast<posit8_2>(start + i * delta);
			}
			return 0;
		}

		void NPyPosit8_2_DotFunc(void *ip1, npy_intp is1, void *ip2, npy_intp is2,
								 void *op, npy_intp n, void *arr)
		{
			char *c1 = reinterpret_cast<char *>(ip1);
			char *c2 = reinterpret_cast<char *>(ip2);
			float acc = 0.0f;
			for (npy_intp i = 0; i < n; ++i)
			{
				posit8_2 *const b1 = reinterpret_cast<posit8_2 *>(c1);
				posit8_2 *const b2 = reinterpret_cast<posit8_2 *>(c2);
				acc += static_cast<float>(*b1) * static_cast<float>(*b2);
				c1 += is1;
				c2 += is2;
			}
			posit8_2 *out = reinterpret_cast<posit8_2 *>(op);
			*out = static_cast<posit8_2>(acc);
		}

		int NPyPosit8_2_CompareFunc(const void *v1, const void *v2, void *arr)
		{
#ifdef DEBUG_CALLS
			std::cout << "NPyPosit8_2_CompareFunc\n";
#endif
			posit8_2 b1 = *reinterpret_cast<const posit8_2 *>(v1);
			posit8_2 b2 = *reinterpret_cast<const posit8_2 *>(v2);
			if (b1 < b2)
			{
				return -1;
			}
			if (b1 > b2)
			{
				return 1;
			}
			if (!Eigen::numext::isnan(b1) && Eigen::numext::isnan(b2))
			{
				return 1;
			}
			if (Eigen::numext::isnan(b2) && !Eigen::numext::isnan(b1))
			{
				return -1;
			}
			return 0;
		}

		int NPyPosit8_2_ArgMaxFunc(void *data, npy_intp n, npy_intp *max_ind,
								   void *arr)
		{
			const posit8_2 *bdata = reinterpret_cast<const posit8_2 *>(data);
			float max_val = -std::numeric_limits<float>::infinity();
			for (npy_intp i = 0; i < n; ++i)
			{
				if (static_cast<float>(bdata[i]) > max_val)
				{
					max_val = static_cast<float>(bdata[i]);
					*max_ind = i;
				}
			}
			return 0;
		}

		int NPyPosit8_2_ArgMinFunc(void *data, npy_intp n, npy_intp *min_ind,
								   void *arr)
		{
			const posit8_2 *bdata = reinterpret_cast<const posit8_2 *>(data);
			float min_val = std::numeric_limits<float>::infinity();
			for (npy_intp i = 0; i < n; ++i)
			{
				if (static_cast<float>(bdata[i]) < min_val)
				{
					min_val = static_cast<float>(bdata[i]);
					*min_ind = i;
				}
			}
			return 0;
		}

		// NumPy casts
		template <typename T, typename Enable = void>
		struct TypeDescriptor
		{
			// typedef ... T;  // Representation type in memory for NumPy values of type
			// static int Dtype() { return NPY_...; }  // Numpy type number for T.
		};

		template <>
		struct TypeDescriptor<posit8_2>
		{
			typedef posit8_2 T;
			static int Dtype() { return npy_posit8_2; }
		};

		template <>
		struct TypeDescriptor<uint8>
		{
			typedef uint8 T;
			static int Dtype() { return NPY_UINT8; }
		};

		template <>
		struct TypeDescriptor<uint16>
		{
			typedef uint16 T;
			static int Dtype() { return NPY_UINT16; }
		};

		// We register "int", "long", and "long long" types for portability across
		// Linux, where "int" and "long" are the same type, and Windows, where "long"
		// and "longlong" are the same type.
		template <>
		struct TypeDescriptor<unsigned int>
		{
			typedef unsigned int T;
			static int Dtype() { return NPY_UINT; }
		};

		template <>
		struct TypeDescriptor<unsigned long>
		{							 // NOLINT
			typedef unsigned long T; // NOLINT
			static int Dtype() { return NPY_ULONG; }
		};

		template <>
		struct TypeDescriptor<unsigned long long>
		{								  // NOLINT
			typedef unsigned long long T; // NOLINT
			static int Dtype() { return NPY_ULONGLONG; }
		};

		template <>
		struct TypeDescriptor<int8>
		{
			typedef int8 T;
			static int Dtype() { return NPY_INT8; }
		};

		template <>
		struct TypeDescriptor<int16>
		{
			typedef int16 T;
			static int Dtype() { return NPY_INT16; }
		};

		template <>
		struct TypeDescriptor<int>
		{
			typedef int T;
			static int Dtype() { return NPY_INT; }
		};

		template <>
		struct TypeDescriptor<long>
		{					// NOLINT
			typedef long T; // NOLINT
			static int Dtype() { return NPY_LONG; }
		};

		template <>
		struct TypeDescriptor<long long>
		{						 // NOLINT
			typedef long long T; // NOLINT
			static int Dtype() { return NPY_LONGLONG; }
		};

		template <>
		struct TypeDescriptor<bool>
		{
			typedef int8 T;
			static int Dtype() { return NPY_BOOL; }
		};

		//template <>
		//struct TypeDescriptor<Eigen::half>
		//{
		//	typedef Eigen::half T;
		//	static int Dtype() { return NPY_HALF; }
		//};

		template <>
		struct TypeDescriptor<float>
		{
			typedef float T;
			static int Dtype() { return NPY_FLOAT; }
		};

		template <>
		struct TypeDescriptor<double>
		{
			typedef double T;
			static int Dtype() { return NPY_DOUBLE; }
		};

		//template <>
		//struct TypeDescriptor<std::complex<float>>
		//{
		//	typedef std::complex<float> T;
		//	static int Dtype() { return NPY_COMPLEX64; }
		//};

		//template <>
		//struct TypeDescriptor<std::complex<double>>
		//{
		//	typedef std::complex<double> T;
		//	static int Dtype() { return NPY_COMPLEX128; }
		//};

		template <>
		struct TypeDescriptor<PyObject *>
		{
			typedef void * T;
			static int Dtype() { return NPY_OBJECT; }
		};

		// Performs a NumPy array cast from type 'From' to 'To'.
		template <typename From, typename To>
		void NPyCast(void *from_void, void *to_void, npy_intp n, void *fromarr,
					 void *toarr)
		{
			//sw::universal::bitblock<8UL> raw = p.get();
			
			const auto *from =
				reinterpret_cast<typename TypeDescriptor<From>::T *>(from_void);
			auto *to = reinterpret_cast<typename TypeDescriptor<To>::T *>(to_void);
			// using const index, 0, not good idea
			const char* positType1 = "posit<8, 2>";//"sw::universal::posit<8, 2>";
			const char* positType2 = "const posit<8, 2>";//"const sw::universal::posit<8, 2>";
			for (npy_intp i = 0; i < n; ++i)
			{
				const char* fromType = typeid(from[i]).name();
				const char* toType = typeid(to[i]).name();
				
				//for conversion of other type to other type
				if (
					(strcmp(fromType, positType1)!=0 || strcmp(fromType, positType2)!=0)
					&&
					(strcmp(toType, positType1)!=0 || strcmp(toType, positType2)!=0)
				) {
					to[i] = static_cast<typename TypeDescriptor<To>::T>(static_cast<To>(from[i]));	
				}
				// convert from posit<8,2> to other types - cast operators
				//else if (fromType.find(positType) != std::string::npos) {	
				else if (strcmp(fromType, positType1)==0 || strcmp(fromType, positType2)==0) {
					to[i] = static_cast<typename TypeDescriptor<To>::T>(static_cast<To>(from[i]));
				} 
				// convert from other types to posit<8,2> - use constructors
				else if (strcmp(toType, positType1)==0 || strcmp(toType, positType2)==0) {
					sw::universal::posit<8UL, 2UL> pa(from[i]);
					to[i] = static_cast<To>(pa);
					//static_cast<typename TypeDescriptor<To>::T>(static_cast<To>(from[i]));
				}
			}
			
			//	to[i] =
			//		static_cast<typename TypeDescriptor<To>::T>(static_cast<To>(from[i].get()));
		}

		// Registers a cast between posit8_2 and type 'T'. 'numpy_type' is the NumPy
		// type corresponding to 'T'. If 'cast_is_safe', registers that posit8_2 can be
		// safely coerced to T.
		template <typename T>
		bool RegisterPosit8_2Cast(int numpy_type, bool cast_is_safe)
		{
			if (PyArray_RegisterCastFunc(PyArray_DescrFromType(numpy_type), npy_posit8_2, NPyCast<T, posit8_2>) < 0)
			{
				return false;
			}
			if (PyArray_RegisterCastFunc(&NPyPosit8_2_Descr, numpy_type, NPyCast<posit8_2, T>) < 0)
			{
				return false;
			}
			if (cast_is_safe && PyArray_RegisterCanCast(&NPyPosit8_2_Descr, numpy_type, NPY_NOSCALAR) < 0)
			{
				return false;
			}
			return true;
		}

		template <typename InType, typename OutType, typename Functor>
		struct UnaryUFunc
		{
			static std::vector<int> Types()
			{
				return {TypeDescriptor<InType>::Dtype(), TypeDescriptor<OutType>::Dtype()};
			}
			static void Call(char **args, const npy_intp *dimensions,
							 const npy_intp *steps, void *data)
			{
				const char *i0 = args[0];
				char *o = args[1];
				for (npy_intp k = 0; k < *dimensions; k++)
				{
					auto x = *reinterpret_cast<const typename TypeDescriptor<InType>::T *>(i0);
					*reinterpret_cast<typename TypeDescriptor<OutType>::T *>(o) = Functor()(x);
					i0 += steps[0];
					o += steps[1];
				}
			}
		};

		template <typename InType, typename OutType, typename OutType2,
				  typename Functor>
		struct UnaryUFunc2
		{
			static std::vector<int> Types()
			{
				return {TypeDescriptor<InType>::Dtype(), TypeDescriptor<OutType>::Dtype(),
						TypeDescriptor<OutType2>::Dtype()};
			}
			static void Call(char **args, const npy_intp *dimensions,
							 const npy_intp *steps, void *data)
			{
				const char *i0 = args[0];
				char *o0 = args[1];
				char *o1 = args[2];
				for (npy_intp k = 0; k < *dimensions; k++)
				{
					auto x = *reinterpret_cast<const typename TypeDescriptor<InType>::T *>(i0);
					std::tie(*reinterpret_cast<typename TypeDescriptor<OutType>::T *>(o0),
							 *reinterpret_cast<typename TypeDescriptor<OutType2>::T *>(o1)) =
						Functor()(x);
					i0 += steps[0];
					o0 += steps[1];
					o1 += steps[2];
				}
			}
		};

		template <typename InType, typename OutType, typename Functor>
		struct BinaryUFunc
		{
			static std::vector<int> Types()
			{
				return {TypeDescriptor<InType>::Dtype(), TypeDescriptor<InType>::Dtype(),
						TypeDescriptor<OutType>::Dtype()};
			}
			static void Call(char **args, const npy_intp *dimensions,
							 const npy_intp *steps, void *data)
			{
#ifdef DEBUG_CALLS
				std::cout << "BinaryUFunc->Call\n";
#endif
				const char *i0 = args[0];
				const char *i1 = args[1];
				char *o = args[2];
				fenv_t fenv;
				feholdexcept(&fenv);
				for (npy_intp k = 0; k < *dimensions; k++)
				{
					auto x = *reinterpret_cast<const typename TypeDescriptor<InType>::T *>(i0);
					auto y = *reinterpret_cast<const typename TypeDescriptor<InType>::T *>(i1);
					*reinterpret_cast<typename TypeDescriptor<OutType>::T *>(o) =
						Functor()(x, y);
					i0 += steps[0];
					i1 += steps[1];
					o += steps[2];
				}
				if (fetestexcept(FE_INVALID | FE_DIVBYZERO | FE_OVERFLOW | FE_UNDERFLOW)) {
					if (fetestexcept(FE_INVALID)) {
						PyErr_SetString(PyExc_ArithmeticError, "posit8_2 invalid");
					} else if (fetestexcept(FE_DIVBYZERO)) {
						PyErr_SetString(PyExc_ArithmeticError, "posit8_2 divide by zero");
					} else if (fetestexcept(FE_OVERFLOW)) {
						PyErr_SetString(PyExc_ArithmeticError, "posit8_2 overflow");
					} else if (fetestexcept(FE_UNDERFLOW)) {
						PyErr_SetString(PyExc_ArithmeticError, "posit8_2 underflow");
					}
				}
				fesetenv(&fenv);
			}
		};

		template <typename InType, typename InType2, typename OutType, typename Functor>
		struct BinaryUFunc2
		{
			static std::vector<int> Types()
			{
				return {TypeDescriptor<InType>::Dtype(), TypeDescriptor<InType2>::Dtype(),
						TypeDescriptor<OutType>::Dtype()};
			}
			static void Call(char **args, const npy_intp *dimensions,
							 const npy_intp *steps, void *data)
			{
#ifdef DEBUG_CALLS
				std::cout << "BinaryUFunc2->Call\n";
#endif
				const char *i0 = args[0];
				const char *i1 = args[1];
				char *o = args[2];
				fenv_t fenv;
				feholdexcept(&fenv);
				for (npy_intp k = 0; k < *dimensions; k++)
				{
					auto x = *reinterpret_cast<const typename TypeDescriptor<InType>::T *>(i0);
					auto y =
						*reinterpret_cast<const typename TypeDescriptor<InType2>::T *>(i1);
					*reinterpret_cast<typename TypeDescriptor<OutType>::T *>(o) =
						Functor()(x, y);
					i0 += steps[0];
					i1 += steps[1];
					o += steps[2];
				}
				if (fetestexcept(FE_INVALID | FE_DIVBYZERO | FE_OVERFLOW | FE_UNDERFLOW)) {
					if (fetestexcept(FE_INVALID)) {
						PyErr_SetString(PyExc_ArithmeticError, "posit8_2 invalid");
					} else if (fetestexcept(FE_DIVBYZERO)) {
						PyErr_SetString(PyExc_ArithmeticError, "posit8_2 divide by zero");
					} else if (fetestexcept(FE_OVERFLOW)) {
						PyErr_SetString(PyExc_ArithmeticError, "posit8_2 overflow");
					} else if (fetestexcept(FE_UNDERFLOW)) {
						PyErr_SetString(PyExc_ArithmeticError, "posit8_2 underflow");
					}
				}
				fesetenv(&fenv);
			}
		};

		// template <typename InType, typename OutType, typename Functor>
		// struct BinaryUFuncObj
		// {
		// 	static std::vector<int> Types()
		// 	{
		// 		return {TypeDescriptor<InType>::Dtype(), NPY_OBJECT,
		// 				TypeDescriptor<OutType>::Dtype()};
		// 	}
		// 	static void Call(char **args, const npy_intp *dimensions,
		// 					 const npy_intp *steps, void *data)
		// 	{
		// 		const char *i0 = args[0];
		// 		char *i1 = args[1];
		// 		char *o = args[2];
		// 		for (npy_intp k = 0; k < *dimensions; k++)
		// 		{
		// 			auto x = *reinterpret_cast<const typename TypeDescriptor<InType>::T *>(i0);
		// 			posit8_2 y = *reinterpret_cast<posit8_2 *>(i1);
		// 			*reinterpret_cast<typename TypeDescriptor<OutType>::T *>(o) =
		// 				Functor()(x, y);
		// 			i0 += steps[0];
		// 			i1 += steps[1];
		// 			o += steps[2];
		// 		}
		// 	}
		// };

		template <typename UFunc>
		bool RegisterUFunc(PyObject *numpy, const char *name)
		{
			std::vector<int> types = UFunc::Types();
			PyUFuncGenericFunction fn =
				reinterpret_cast<PyUFuncGenericFunction>(UFunc::Call);
			Safe_PyObjectPtr ufunc_obj = make_safe(PyObject_GetAttrString(numpy, name));
			if (!ufunc_obj)
			{
				return false;
			}
			PyUFuncObject *ufunc = reinterpret_cast<PyUFuncObject *>(ufunc_obj.get());
			if (static_cast<int>(types.size()) != ufunc->nargs)
			{
				PyErr_Format(PyExc_AssertionError,
							 "ufunc %s takes %d arguments, loop takes %lu", name,
							 ufunc->nargs, types.size());
				return false;
			}
			if (PyUFunc_RegisterLoopForType(ufunc, npy_posit8_2, fn,
											const_cast<int *>(types.data()),
											nullptr) < 0)
			{
				return false;
			}
			return true;
		}

		namespace ufuncs
		{

			struct Add
			{
				posit8_2 operator()(posit8_2 a, posit8_2 b) { return a + b; }
			};
			struct AddScalarFloat
			{
				posit8_2 operator()(posit8_2 a, float b) { return a + posit8_2(b); }
			};
			struct ScalarFloatAdd
			{
				posit8_2 operator()(float a, posit8_2 b) { return posit8_2(a) + b; }
			};
			struct Subtract
			{
				posit8_2 operator()(posit8_2 a, posit8_2 b) { return a - b; }
			};
			struct Multiply
			{
				posit8_2 operator()(posit8_2 a, posit8_2 b) { return a * b; }
			};
			struct TrueDivide
			{
				posit8_2 operator()(posit8_2 a, posit8_2 b) { return a / b; }
			};

			std::pair<float, float> divmod(float a, float b)
			{
				if (b == 0.0f)
				{
					float nan = std::numeric_limits<float>::quiet_NaN();
					return {nan, nan};
				}
				float mod = std::fmod(a, b);
				float div = (a - mod) / b;
				if (mod != 0.0f)
				{
					if ((b < 0.0f) != (mod < 0.0f))
					{
						mod += b;
						div -= 1.0f;
					}
				}
				else
				{
					mod = std::copysign(0.0f, b);
				}

				float floordiv;
				if (div != 0.0f)
				{
					floordiv = std::floor(div);
					if (div - floordiv > 0.5f)
					{
						floordiv += 1.0f;
					}
				}
				else
				{
					floordiv = std::copysign(0.0f, a / b);
				}
				return {floordiv, mod};
			}

			struct FloorDivide
			{
				posit8_2 operator()(posit8_2 a, posit8_2 b)
				{
					return posit8_2(divmod(static_cast<float>(a), static_cast<float>(b)).first);
				}
			};
			struct Remainder
			{
				posit8_2 operator()(posit8_2 a, posit8_2 b)
				{
					return posit8_2(
						divmod(static_cast<float>(a), static_cast<float>(b)).second);
				}
			};
			struct DivmodUFunc
			{
				static std::vector<int> Types()
				{
					return {npy_posit8_2, npy_posit8_2, npy_posit8_2, npy_posit8_2};
				}
				static void Call(char **args, npy_intp *dimensions, npy_intp *steps,
								 void *data)
				{
					const char *i0 = args[0];
					const char *i1 = args[1];
					char *o0 = args[2];
					char *o1 = args[3];
					for (npy_intp k = 0; k < *dimensions; k++)
					{
						posit8_2 x = *reinterpret_cast<const posit8_2 *>(i0);
						posit8_2 y = *reinterpret_cast<const posit8_2 *>(i1);
						float floordiv, mod;
						std::tie(floordiv, mod) =
							divmod(static_cast<float>(x), static_cast<float>(y));
						*reinterpret_cast<posit8_2 *>(o0) = posit8_2(floordiv);
						*reinterpret_cast<posit8_2 *>(o1) = posit8_2(mod);
						i0 += steps[0];
						i1 += steps[1];
						o0 += steps[2];
						o1 += steps[3];
					}
				}
			};
			struct Fmod
			{
				posit8_2 operator()(posit8_2 a, posit8_2 b)
				{
					return posit8_2(std::fmod(static_cast<float>(a), static_cast<float>(b)));
				}
			};
			struct Negative
			{
				posit8_2 operator()(posit8_2 a) { return -a; }
			};
			struct Positive
			{
				posit8_2 operator()(posit8_2 a) { return a; }
			};
			struct Power
			{
				posit8_2 operator()(posit8_2 a, posit8_2 b)
				{
					return posit8_2(std::pow(static_cast<float>(a), static_cast<float>(b)));
				}
			};
			struct Abs
			{
				posit8_2 operator()(posit8_2 a)
				{
					return posit8_2(std::abs(static_cast<float>(a)));
				}
			};
			struct Cbrt
			{
				posit8_2 operator()(posit8_2 a)
				{
					return posit8_2(std::cbrt(static_cast<float>(a)));
				}
			};
			struct Ceil
			{
				posit8_2 operator()(posit8_2 a)
				{
					return posit8_2(std::ceil(static_cast<float>(a)));
				}
			};
			struct CopySign
			{
				posit8_2 operator()(posit8_2 a, posit8_2 b)
				{
					return posit8_2(
						std::copysign(static_cast<float>(a), static_cast<float>(b)));
				}
			};
			struct Exp
			{
				posit8_2 operator()(posit8_2 a)
				{
					return posit8_2(std::exp(static_cast<float>(a)));
				}
			};
			struct Exp2
			{
				posit8_2 operator()(posit8_2 a)
				{
					return posit8_2(std::exp2(static_cast<float>(a)));
				}
			};
			struct Expm1
			{
				posit8_2 operator()(posit8_2 a)
				{
					return posit8_2(std::expm1(static_cast<float>(a)));
				}
			};
			struct Floor
			{
				posit8_2 operator()(posit8_2 a)
				{
					return posit8_2(std::floor(static_cast<float>(a)));
				}
			};
			struct Frexp
			{
				std::pair<posit8_2, int> operator()(posit8_2 a)
				{
					int exp;
					float f = std::frexp(static_cast<float>(a), &exp);
					return {posit8_2(f), exp};
				}
			};
			struct Heaviside
			{
				posit8_2 operator()(posit8_2 bx, posit8_2 h0)
				{
					float x = static_cast<float>(bx);
					if (Eigen::numext::isnan(x))
					{
						return bx;
					}
					if (x < 0)
					{
						return posit8_2(0.0f);
					}
					if (x > 0)
					{
						return posit8_2(1.0f);
					}
					return h0; // x == 0
				}
			};
			struct Conjugate
			{
				posit8_2 operator()(posit8_2 a) { return a; }
			};
			struct IsFinite
			{
				bool operator()(posit8_2 a) { return std::isfinite(static_cast<float>(a)); }
			};
			struct IsInf
			{
				bool operator()(posit8_2 a) { return std::isinf(static_cast<float>(a)); }
			};
			struct IsNan
			{
				bool operator()(posit8_2 a)
				{
					return Eigen::numext::isnan(static_cast<float>(a));
				}
			};
			struct Ldexp
			{
				posit8_2 operator()(posit8_2 a, int exp)
				{
					return posit8_2(std::ldexp(static_cast<float>(a), exp));
				}
			};
			struct Log
			{
				posit8_2 operator()(posit8_2 a)
				{
					return posit8_2(std::log(static_cast<float>(a)));
				}
			};
			struct Log2
			{
				posit8_2 operator()(posit8_2 a)
				{
					return posit8_2(std::log2(static_cast<float>(a)));
				}
			};
			struct Log10
			{
				posit8_2 operator()(posit8_2 a)
				{
					return posit8_2(std::log10(static_cast<float>(a)));
				}
			};
			struct Log1p
			{
				posit8_2 operator()(posit8_2 a)
				{
					return posit8_2(std::log1p(static_cast<float>(a)));
				}
			};
			struct LogAddExp
			{
				posit8_2 operator()(posit8_2 bx, posit8_2 by)
				{
					float x = static_cast<float>(bx);
					float y = static_cast<float>(by);
					if (x == y)
					{
						// Handles infinities of the same sign.
						return posit8_2(x + std::log(2.0f));
					}
					float out = std::numeric_limits<float>::quiet_NaN();
					if (x > y)
					{
						out = x + std::log1p(std::exp(y - x));
					}
					else if (x < y)
					{
						out = y + std::log1p(std::exp(x - y));
					}
					return posit8_2(out);
				}
			};
			struct LogAddExp2
			{
				posit8_2 operator()(posit8_2 bx, posit8_2 by)
				{
					float x = static_cast<float>(bx);
					float y = static_cast<float>(by);
					if (x == y)
					{
						// Handles infinities of the same sign.
						return posit8_2(x + 1.0f);
					}
					float out = std::numeric_limits<float>::quiet_NaN();
					if (x > y)
					{
						out = x + std::log1p(std::exp2(y - x)) / std::log(2.0f);
					}
					else if (x < y)
					{
						out = y + std::log1p(std::exp2(x - y)) / std::log(2.0f);
					}
					return posit8_2(out);
				}
			};
			struct Modf
			{
				std::pair<posit8_2, posit8_2> operator()(posit8_2 a)
				{
					float integral;
					float f = std::modf(static_cast<float>(a), &integral);
					return {posit8_2(f), posit8_2(integral)};
				}
			};

			struct Reciprocal
			{
				posit8_2 operator()(posit8_2 a)
				{
					return posit8_2(1.f / static_cast<float>(a));
				}
			};
			struct Rint
			{
				posit8_2 operator()(posit8_2 a)
				{
					return posit8_2(std::rint(static_cast<float>(a)));
				}
			};
			struct Sign
			{
				posit8_2 operator()(posit8_2 a)
				{
					float f(a);
					if (f < 0)
					{
						return posit8_2(-1);
					}
					if (f > 0)
					{
						return posit8_2(1);
					}
					return a;
				}
			};
			struct SignBit
			{
				bool operator()(posit8_2 a) { return std::signbit(static_cast<float>(a)); }
			};
			struct Sqrt
			{
				posit8_2 operator()(posit8_2 a)
				{
					return posit8_2(std::sqrt(static_cast<float>(a)));
				}
			};
			struct Square
			{
				posit8_2 operator()(posit8_2 a)
				{
					float f(a);
					return posit8_2(f * f);
				}
			};
			struct Trunc
			{
				posit8_2 operator()(posit8_2 a)
				{
					return posit8_2(std::trunc(static_cast<float>(a)));
				}
			};

			// Trigonometric functions
			struct Sin
			{
				posit8_2 operator()(posit8_2 a)
				{
					return posit8_2(std::sin(static_cast<float>(a)));
				}
			};
			struct Cos
			{
				posit8_2 operator()(posit8_2 a)
				{
					return posit8_2(std::cos(static_cast<float>(a)));
				}
			};
			struct Tan
			{
				posit8_2 operator()(posit8_2 a)
				{
					return posit8_2(std::tan(static_cast<float>(a)));
				}
			};
			struct Arcsin
			{
				posit8_2 operator()(posit8_2 a)
				{
					return posit8_2(std::asin(static_cast<float>(a)));
				}
			};
			struct Arccos
			{
				posit8_2 operator()(posit8_2 a)
				{
					return posit8_2(std::acos(static_cast<float>(a)));
				}
			};
			struct Arctan
			{
				posit8_2 operator()(posit8_2 a)
				{
					return posit8_2(std::atan(static_cast<float>(a)));
				}
			};
			struct Arctan2
			{
				posit8_2 operator()(posit8_2 a, posit8_2 b)
				{
					return posit8_2(std::atan2(static_cast<float>(a), static_cast<float>(b)));
				}
			};
			struct Hypot
			{
				posit8_2 operator()(posit8_2 a, posit8_2 b)
				{
					return posit8_2(std::hypot(static_cast<float>(a), static_cast<float>(b)));
				}
			};
			struct Sinh
			{
				posit8_2 operator()(posit8_2 a)
				{
					return posit8_2(std::sinh(static_cast<float>(a)));
				}
			};
			struct Cosh
			{
				posit8_2 operator()(posit8_2 a)
				{
					return posit8_2(std::cosh(static_cast<float>(a)));
				}
			};
			struct Tanh
			{
				posit8_2 operator()(posit8_2 a)
				{
					return posit8_2(std::tanh(static_cast<float>(a)));
				}
			};
			struct Arcsinh
			{
				posit8_2 operator()(posit8_2 a)
				{
					return posit8_2(std::asinh(static_cast<float>(a)));
				}
			};
			struct Arccosh
			{
				posit8_2 operator()(posit8_2 a)
				{
					return posit8_2(std::acosh(static_cast<float>(a)));
				}
			};
			struct Arctanh
			{
				posit8_2 operator()(posit8_2 a)
				{
					return posit8_2(std::atanh(static_cast<float>(a)));
				}
			};
			struct Deg2rad
			{
				posit8_2 operator()(posit8_2 a)
				{
					static constexpr float radians_per_degree = M_PI / 180.0f;
					return posit8_2(static_cast<float>(a) * radians_per_degree);
				}
			};
			struct Rad2deg
			{
				posit8_2 operator()(posit8_2 a)
				{
					static constexpr float degrees_per_radian = 180.0f / M_PI;
					return posit8_2(static_cast<float>(a) * degrees_per_radian);
				}
			};

			struct Eq
			{
				npy_bool operator()(posit8_2 a, posit8_2 b) { return a == b; }
			};
			struct EqFloat
			{
				npy_bool operator()(posit8_2 a, float b) { return a == posit8_2(b); }
			};
			struct EqDouble
			{
				npy_bool operator()(posit8_2 a, double b) { return a == posit8_2(b); }
			};
			struct Ne
			{
				npy_bool operator()(posit8_2 a, posit8_2 b) { return a != b; }
			};
			struct NeFloat
			{
				npy_bool operator()(posit8_2 a, float b) { return a != posit8_2(b); }
			};
			struct NeDouble
			{
				npy_bool operator()(posit8_2 a, double b) { return a != posit8_2(b); }
			};
			struct Lt
			{
				npy_bool operator()(posit8_2 a, posit8_2 b) { return a < b; }
			};
			struct LtFloat
			{
				npy_bool operator()(posit8_2 a, float b) { return a < posit8_2(b); }
			};
			struct LtDouble
			{
				npy_bool operator()(posit8_2 a, double b) { return a < posit8_2(b); }
			};
			struct Gt
			{
				npy_bool operator()(posit8_2 a, posit8_2 b) { return a > b; }
			};
			struct GtFloat
			{
				npy_bool operator()(posit8_2 a, float b) { return a > posit8_2(b); }
			};
			struct GtDouble
			{
				npy_bool operator()(posit8_2 a, double b) { return a > posit8_2(b); }
			};
			struct Le
			{
				npy_bool operator()(posit8_2 a, posit8_2 b) { return a <= b; }
			};
			struct LeFloat
			{
				npy_bool operator()(posit8_2 a, float b) { return a <= posit8_2(b); }
			};
			struct LeDouble
			{
				npy_bool operator()(posit8_2 a, double b) { return a <= posit8_2(b); }
			};
			struct Ge
			{
				npy_bool operator()(posit8_2 a, posit8_2 b) { return a >= b; }
			};
			struct GeFloat
			{
				npy_bool operator()(posit8_2 a, float b) { return a >= posit8_2(b); }
			};
			struct GeDouble
			{
				npy_bool operator()(posit8_2 a, double b) { return a >= posit8_2(b); }
			};
			struct Maximum
			{
				posit8_2 operator()(posit8_2 a, posit8_2 b)
				{
					float fa(a), fb(b);
					return Eigen::numext::isnan(fa) || fa > fb ? a : b;
				}
			};
			struct Minimum
			{
				posit8_2 operator()(posit8_2 a, posit8_2 b)
				{
					float fa(a), fb(b);
					return Eigen::numext::isnan(fa) || fa < fb ? a : b;
				}
			};
			struct Fmax
			{
				posit8_2 operator()(posit8_2 a, posit8_2 b)
				{
					float fa(a), fb(b);
					return Eigen::numext::isnan(fb) || fa > fb ? a : b;
				}
			};
			struct Fmin
			{
				posit8_2 operator()(posit8_2 a, posit8_2 b)
				{
					float fa(a), fb(b);
					return Eigen::numext::isnan(fb) || fa < fb ? a : b;
				}
			};

/*			struct LogicalNot
			{
				npy_bool operator()(posit8_2 a) { return !a; }
			};
			struct LogicalAnd
			{
				npy_bool operator()(posit8_2 a, posit8_2 b) { return a && b; }
			};
			struct LogicalOr
			{
				npy_bool operator()(posit8_2 a, posit8_2 b) { return a || b; }
			};
			struct LogicalXor
			{
				npy_bool operator()(posit8_2 a, posit8_2 b)
				{
					return static_cast<bool>(a) ^ static_cast<bool>(b);
				}
			};
*/
			struct NextAfter
			{
				posit8_2 operator()(posit8_2 from, posit8_2 to)
				{
					long from_as_int, to_as_int;
					const long sign_mask = 1 << 7;
					float from_as_float(from), to_as_float(to);
					memcpy(&from_as_int, &from, sizeof(posit8_2));
					memcpy(&to_as_int, &to, sizeof(posit8_2));
					if (Eigen::numext::isnan(from_as_float) ||
						Eigen::numext::isnan(to_as_float))
					{
						return posit8_2(std::numeric_limits<float>::quiet_NaN());
					}
					if (from_as_int == to_as_int)
					{
						return to;
					}
					if (from_as_float == 0)
					{
						if (to_as_float == 0)
						{
							return to;
						}
						else
						{
							// Smallest subnormal signed like `to`.
							uint8_t out_int = (to_as_int & sign_mask) | 1;
							posit8_2 out;
							memcpy(&out, &out_int, sizeof(posit8_2));
							return out;
						}
					}
					long from_sign = from_as_int & sign_mask;
					long to_sign = to_as_int & sign_mask;
					long from_abs = from_as_int & ~sign_mask;
					long to_abs = to_as_int & ~sign_mask;
					long magnitude_adjustment =
						(from_abs > to_abs || from_sign != to_sign) ? 0xFFFF : 0x0001;
					long out_int = from_as_int + magnitude_adjustment;
					posit8_2 out;
					memcpy(&out, &out_int, sizeof(posit8_2));
					return out;
				}
			};
			// TODO(phawkins): implement spacing

		} // namespace ufuncs

	} // namespace

	// Initializes the module.
	bool Initialize()
	{
		import_array();
		import_umath1(false);

		Safe_PyObjectPtr numpy_str = make_safe(PyUnicode_FromString("numpy"));
		if (!numpy_str)
		{
			return false;
		}
		Safe_PyObjectPtr numpy = make_safe(PyImport_Import(numpy_str.get()));
		if (!numpy)
		{
			return false;
		}

		// If another module (presumably either TF or JAX) has registered a posit8_2
		// type, use it. We don't want two posit8_2 types if we can avoid it since it
		// leads to confusion if we have two different types with the same name. This
		// assumes that the other module has a sufficiently complete posit8_2
		// implementation. The only known NumPy posit8_2 extension at the time of
		// writing is this one (distributed in TF and JAX).
		// TODO(phawkins): distribute the posit8_2 extension as its own pip package,
		// so we can unambiguously refer to a single canonical definition of posit8_2.
		int typenum = PyArray_TypeNumFromName(const_cast<char *>("posit<8,2>"));
		if (typenum != NPY_NOTYPE)
		{
			PyArray_Descr *descr = PyArray_DescrFromType(typenum);
			// The test for an argmax function here is to verify that the
			// posit8_2 implementation is sufficiently new, and, say, not from
			// an older version of TF or JAX.
			if (descr && descr->f && descr->f->argmax)
			{
				npy_posit8_2 = typenum;
				posit8_2_type_ptr = descr->typeobj;
				return true;
			}
		}

		posit8_2_type.tp_base = &PyGenericArrType_Type;

		if (PyType_Ready(&posit8_2_type) < 0)
		{
			PyErr_Print();
	        PyErr_SetString(PyExc_SystemError, "could not initialize posit8_2");
			return false;
		}

		// Initializes the NumPy descriptor.
		PyArray_InitArrFuncs(&NPyPosit8_2_ArrFuncs);
		NPyPosit8_2_ArrFuncs.getitem = NPyPosit8_2_GetItem;
		NPyPosit8_2_ArrFuncs.setitem = NPyPosit8_2_SetItem;
		NPyPosit8_2_ArrFuncs.copyswapn = NPyPosit8_2_CopySwapN;
		NPyPosit8_2_ArrFuncs.copyswap = NPyPosit8_2_CopySwap;
		NPyPosit8_2_ArrFuncs.nonzero = NPyPosit8_2_NonZero;
		NPyPosit8_2_ArrFuncs.fill = NPyPosit8_2_Fill;
		NPyPosit8_2_ArrFuncs.dotfunc = NPyPosit8_2_DotFunc;
		NPyPosit8_2_ArrFuncs.compare = NPyPosit8_2_CompareFunc;
		NPyPosit8_2_ArrFuncs.argmax = NPyPosit8_2_ArgMaxFunc;
		NPyPosit8_2_ArrFuncs.argmin = NPyPosit8_2_ArgMinFunc;

		Py_TYPE(&NPyPosit8_2_Descr) = &PyArrayDescr_Type;
		npy_posit8_2 = PyArray_RegisterDataType(&NPyPosit8_2_Descr);
		posit8_2_type_ptr = &posit8_2_type;
		if (npy_posit8_2 < 0)
		{
			return false;
		}

		// Support dtype(posit8_2)
		if (PyDict_SetItemString(posit8_2_type.tp_dict, "dtype",
								 reinterpret_cast<PyObject *>(&NPyPosit8_2_Descr)) <
			0)
		{
			return false;
		}

		// Register casts
		//if (!RegisterPosit8_2Cast<Eigen::half>(NPY_HALF, /*cast_is_safe=*/false))
		//{
		//	return false;
		//}
		if (!RegisterPosit8_2Cast<float>(NPY_FLOAT, /*cast_is_safe=*/true))
		{
			return false;
		}
		if (!RegisterPosit8_2Cast<double>(NPY_DOUBLE, /*cast_is_safe=*/true))
		{
			return false;
		}
		//if (!RegisterPosit8_2Cast<bool>(NPY_BOOL, /*cast_is_safe=*/false))
		//{
		//	return false;
		//}
		//if (!RegisterPosit8_2Cast<uint8>(NPY_UINT8, /*cast_is_safe=*/false))
		//{
		//	return false;
		//}
		if (!RegisterPosit8_2Cast<uint16>(NPY_UINT16, /*cast_is_safe=*/false))
		{
			return false;
		}
		if (!RegisterPosit8_2Cast<unsigned int>(NPY_UINT, /*cast_is_safe=*/false))
		{
			return false;
		}
		if (!RegisterPosit8_2Cast<unsigned long>(NPY_ULONG, // NOLINT
												 /*cast_is_safe=*/false))
		{
			return false;
		}
		if (!RegisterPosit8_2Cast<unsigned long long>( // NOLINT
				NPY_ULONGLONG, /*cast_is_safe=*/false))
		{
			return false;
		}
		if (!RegisterPosit8_2Cast<uint64>(NPY_UINT64, /*cast_is_safe=*/false))
		{
			return false;
		}
		//if (!RegisterPosit8_2Cast<int8>(NPY_INT8, /*cast_is_safe=*/false))
		//{
		//	return false;
		//}
		if (!RegisterPosit8_2Cast<int16>(NPY_INT16, /*cast_is_safe=*/false))
		{
			return false;
		}
		if (!RegisterPosit8_2Cast<int>(NPY_INT, /*cast_is_safe=*/false))
		{
			return false;
		}
		if (!RegisterPosit8_2Cast<long>(NPY_LONG, // NOLINT
										/*cast_is_safe=*/false))
		{
			return false;
		}
		if (!RegisterPosit8_2Cast<long long>( // NOLINT
				NPY_LONGLONG, /*cast_is_safe=*/false))
		{
			return false;
		}
		// Following the numpy convention. imag part is dropped when converting to
		// float.
		//if (!RegisterPosit8_2Cast<std::complex<float>>(NPY_COMPLEX64,
		//											   /*cast_is_safe=*/true))
		//{
		//	return false;
		//}
		//if (!RegisterPosit8_2Cast<std::complex<double>>(NPY_COMPLEX128, /*cast_is_safe=*/true))
		//{
		//	return false;
		//}

		bool ok =
			RegisterUFunc<BinaryUFunc<posit8_2, posit8_2, ufuncs::Add>>(numpy.get(), "add") &&
			RegisterUFunc<BinaryUFunc2<float, posit8_2, posit8_2, ufuncs::ScalarFloatAdd>>(numpy.get(), "add") &&
			RegisterUFunc<BinaryUFunc2<posit8_2, float, posit8_2, ufuncs::AddScalarFloat>>(numpy.get(), "add") &&
			RegisterUFunc<BinaryUFunc<posit8_2, posit8_2, ufuncs::Subtract>>(numpy.get(), "subtract") &&
			RegisterUFunc<BinaryUFunc<posit8_2, posit8_2, ufuncs::Multiply>>(
				numpy.get(), "multiply") &&
			RegisterUFunc<BinaryUFunc<posit8_2, posit8_2, ufuncs::TrueDivide>>(
				numpy.get(), "divide") &&
			RegisterUFunc<BinaryUFunc<posit8_2, posit8_2, ufuncs::LogAddExp>>(
				numpy.get(), "logaddexp") &&
			RegisterUFunc<BinaryUFunc<posit8_2, posit8_2, ufuncs::LogAddExp2>>(
				numpy.get(), "logaddexp2") &&
			RegisterUFunc<UnaryUFunc<posit8_2, posit8_2, ufuncs::Negative>>(
				numpy.get(), "negative") &&
			RegisterUFunc<UnaryUFunc<posit8_2, posit8_2, ufuncs::Positive>>(
				numpy.get(), "positive") &&
			RegisterUFunc<BinaryUFunc<posit8_2, posit8_2, ufuncs::TrueDivide>>(
				numpy.get(), "true_divide") &&
			RegisterUFunc<BinaryUFunc<posit8_2, posit8_2, ufuncs::FloorDivide>>(
				numpy.get(), "floor_divide") &&
			RegisterUFunc<BinaryUFunc<posit8_2, posit8_2, ufuncs::Power>>(numpy.get(), "power") &&
			RegisterUFunc<BinaryUFunc<posit8_2, posit8_2, ufuncs::Remainder>>(
				numpy.get(), "remainder") &&
			RegisterUFunc<BinaryUFunc<posit8_2, posit8_2, ufuncs::Remainder>>(
				numpy.get(), "mod") &&
			RegisterUFunc<BinaryUFunc<posit8_2, posit8_2, ufuncs::Fmod>>(numpy.get(), "fmod") &&
			RegisterUFunc<ufuncs::DivmodUFunc>(numpy.get(), "divmod") &&
			RegisterUFunc<UnaryUFunc<posit8_2, posit8_2, ufuncs::Abs>>(numpy.get(), "absolute") &&
			RegisterUFunc<UnaryUFunc<posit8_2, posit8_2, ufuncs::Abs>>(numpy.get(),
																	   "fabs") &&
			RegisterUFunc<UnaryUFunc<posit8_2, posit8_2, ufuncs::Rint>>(numpy.get(),
																		"rint") &&
			RegisterUFunc<UnaryUFunc<posit8_2, posit8_2, ufuncs::Sign>>(numpy.get(),
																		"sign") &&
			RegisterUFunc<BinaryUFunc<posit8_2, posit8_2, ufuncs::Heaviside>>(
				numpy.get(), "heaviside") &&
			RegisterUFunc<UnaryUFunc<posit8_2, posit8_2, ufuncs::Conjugate>>(
				numpy.get(), "conjugate") &&
			RegisterUFunc<UnaryUFunc<posit8_2, posit8_2, ufuncs::Exp>>(numpy.get(),
																	   "exp") &&
			RegisterUFunc<UnaryUFunc<posit8_2, posit8_2, ufuncs::Exp2>>(numpy.get(),
																		"exp2") &&
			RegisterUFunc<UnaryUFunc<posit8_2, posit8_2, ufuncs::Expm1>>(numpy.get(),
																		 "expm1") &&
			RegisterUFunc<UnaryUFunc<posit8_2, posit8_2, ufuncs::Log>>(numpy.get(),
																	   "log") &&
			RegisterUFunc<UnaryUFunc<posit8_2, posit8_2, ufuncs::Log2>>(numpy.get(),
																		"log2") &&
			RegisterUFunc<UnaryUFunc<posit8_2, posit8_2, ufuncs::Log10>>(numpy.get(),
																		 "log10") &&
			RegisterUFunc<UnaryUFunc<posit8_2, posit8_2, ufuncs::Log1p>>(numpy.get(),
																		 "log1p") &&
			RegisterUFunc<UnaryUFunc<posit8_2, posit8_2, ufuncs::Sqrt>>(numpy.get(),
																		"sqrt") &&
			RegisterUFunc<UnaryUFunc<posit8_2, posit8_2, ufuncs::Square>>(numpy.get(),
																		  "square") &&
			RegisterUFunc<UnaryUFunc<posit8_2, posit8_2, ufuncs::Cbrt>>(numpy.get(),
																		"cbrt") &&
			RegisterUFunc<UnaryUFunc<posit8_2, posit8_2, ufuncs::Reciprocal>>(
				numpy.get(), "reciprocal") &&

			// Trigonometric functions
			RegisterUFunc<UnaryUFunc<posit8_2, posit8_2, ufuncs::Sin>>(numpy.get(),
																	   "sin") &&
			RegisterUFunc<UnaryUFunc<posit8_2, posit8_2, ufuncs::Cos>>(numpy.get(),
																	   "cos") &&
			RegisterUFunc<UnaryUFunc<posit8_2, posit8_2, ufuncs::Tan>>(numpy.get(),
																	   "tan") &&
			RegisterUFunc<UnaryUFunc<posit8_2, posit8_2, ufuncs::Arcsin>>(numpy.get(),
																		  "arcsin") &&
			RegisterUFunc<UnaryUFunc<posit8_2, posit8_2, ufuncs::Arccos>>(numpy.get(),
																		  "arccos") &&
			RegisterUFunc<UnaryUFunc<posit8_2, posit8_2, ufuncs::Arctan>>(numpy.get(),
																		  "arctan") &&
			RegisterUFunc<BinaryUFunc<posit8_2, posit8_2, ufuncs::Arctan2>>(
				numpy.get(), "arctan2") &&
			RegisterUFunc<BinaryUFunc<posit8_2, posit8_2, ufuncs::Hypot>>(numpy.get(),
																		  "hypot") &&
			RegisterUFunc<UnaryUFunc<posit8_2, posit8_2, ufuncs::Sinh>>(numpy.get(),
																		"sinh") &&
			RegisterUFunc<UnaryUFunc<posit8_2, posit8_2, ufuncs::Cosh>>(numpy.get(),
																		"cosh") &&
			RegisterUFunc<UnaryUFunc<posit8_2, posit8_2, ufuncs::Tanh>>(numpy.get(),
																		"tanh") &&
			RegisterUFunc<UnaryUFunc<posit8_2, posit8_2, ufuncs::Arcsinh>>(
				numpy.get(), "arcsinh") &&
			RegisterUFunc<UnaryUFunc<posit8_2, posit8_2, ufuncs::Arccosh>>(
				numpy.get(), "arccosh") &&
			RegisterUFunc<UnaryUFunc<posit8_2, posit8_2, ufuncs::Arctanh>>(
				numpy.get(), "arctanh") &&
			RegisterUFunc<UnaryUFunc<posit8_2, posit8_2, ufuncs::Deg2rad>>(
				numpy.get(), "deg2rad") &&
			RegisterUFunc<UnaryUFunc<posit8_2, posit8_2, ufuncs::Rad2deg>>(
				numpy.get(), "rad2deg") &&

			// Comparison functions
			// RegisterUFunc<BinaryUFuncObj<posit8_2, bool, ufuncs::Eq>>(numpy.get(),
			// 													   "equal") &&
			RegisterUFunc<BinaryUFunc<posit8_2, bool, ufuncs::Eq>>(numpy.get(),
																   "equal") &&
			RegisterUFunc<BinaryUFunc2<posit8_2, float, bool, ufuncs::EqFloat>>(numpy.get(),
																    "equal") &&
			RegisterUFunc<BinaryUFunc2<posit8_2, double, bool, ufuncs::EqDouble>>(numpy.get(),
																   "equal") &&
			RegisterUFunc<BinaryUFunc<posit8_2, bool, ufuncs::Ne>>(numpy.get(),
																   "not_equal") &&
			RegisterUFunc<BinaryUFunc2<posit8_2, float, bool, ufuncs::NeFloat>>(numpy.get(),
																   "not_equal") &&
			RegisterUFunc<BinaryUFunc2<posit8_2, double, bool, ufuncs::NeDouble>>(numpy.get(),
																   "not_equal") &&
			RegisterUFunc<BinaryUFunc<posit8_2, bool, ufuncs::Lt>>(numpy.get(),
																   "less") &&
			RegisterUFunc<BinaryUFunc2<posit8_2, float, bool, ufuncs::LtFloat>>(numpy.get(),
																   "less") &&
			RegisterUFunc<BinaryUFunc2<posit8_2, double, bool, ufuncs::LtDouble>>(numpy.get(),
																   "less") &&
			RegisterUFunc<BinaryUFunc<posit8_2, bool, ufuncs::Gt>>(numpy.get(),
																   "greater") &&
			RegisterUFunc<BinaryUFunc2<posit8_2, float, bool, ufuncs::GtFloat>>(numpy.get(),
																   "greater") &&
			RegisterUFunc<BinaryUFunc2<posit8_2, double, bool, ufuncs::GtDouble>>(numpy.get(),
																   "greater") &&
			RegisterUFunc<BinaryUFunc<posit8_2, bool, ufuncs::Le>>(numpy.get(),
																   "less_equal") &&
			RegisterUFunc<BinaryUFunc2<posit8_2, float, bool, ufuncs::LeFloat>>(numpy.get(),
																   "less_equal") &&
			RegisterUFunc<BinaryUFunc2<posit8_2, double, bool, ufuncs::LeDouble>>(numpy.get(),
																   "less_equal") &&
			RegisterUFunc<BinaryUFunc<posit8_2, bool, ufuncs::Ge>>(numpy.get(),
																   "greater_equal") &&
			RegisterUFunc<BinaryUFunc2<posit8_2, float, bool, ufuncs::GeFloat>>(numpy.get(),
																   "greater_equal") &&
			RegisterUFunc<BinaryUFunc2<posit8_2, double, bool, ufuncs::GeDouble>>(numpy.get(),
																   "greater_equal") &&
			RegisterUFunc<BinaryUFunc<posit8_2, posit8_2, ufuncs::Maximum>>(
				numpy.get(), "maximum") &&
			RegisterUFunc<BinaryUFunc<posit8_2, posit8_2, ufuncs::Minimum>>(
				numpy.get(), "minimum") &&
			RegisterUFunc<BinaryUFunc<posit8_2, posit8_2, ufuncs::Fmax>>(numpy.get(),
																		 "fmax") &&
			RegisterUFunc<BinaryUFunc<posit8_2, posit8_2, ufuncs::Fmin>>(numpy.get(),
																		 "fmin") &&
			/*RegisterUFunc<BinaryUFunc<posit8_2, bool, ufuncs::LogicalAnd>>(
				numpy.get(), "logical_and") &&
			RegisterUFunc<BinaryUFunc<posit8_2, bool, ufuncs::LogicalOr>>(
				numpy.get(), "logical_or") &&
			RegisterUFunc<BinaryUFunc<posit8_2, bool, ufuncs::LogicalXor>>(
				numpy.get(), "logical_xor") &&
			RegisterUFunc<UnaryUFunc<posit8_2, bool, ufuncs::LogicalNot>>(
				numpy.get(), "logical_not") &&
			*/
			// Floating point functions
			RegisterUFunc<UnaryUFunc<posit8_2, bool, ufuncs::IsFinite>>(numpy.get(),
																		"isfinite") &&
			RegisterUFunc<UnaryUFunc<posit8_2, bool, ufuncs::IsInf>>(numpy.get(),
																	 "isinf") &&
			RegisterUFunc<UnaryUFunc<posit8_2, bool, ufuncs::IsNan>>(numpy.get(),
																	 "isnan") &&
			RegisterUFunc<UnaryUFunc<posit8_2, bool, ufuncs::SignBit>>(numpy.get(),
																	   "signbit") &&
			RegisterUFunc<BinaryUFunc<posit8_2, posit8_2, ufuncs::CopySign>>(
				numpy.get(), "copysign") &&
			RegisterUFunc<UnaryUFunc2<posit8_2, posit8_2, posit8_2, ufuncs::Modf>>(
				numpy.get(), "modf") &&
			RegisterUFunc<BinaryUFunc2<posit8_2, int, posit8_2, ufuncs::Ldexp>>(
				numpy.get(), "ldexp") &&
			RegisterUFunc<UnaryUFunc2<posit8_2, posit8_2, int, ufuncs::Frexp>>(
				numpy.get(), "frexp") &&
			RegisterUFunc<UnaryUFunc<posit8_2, posit8_2, ufuncs::Floor>>(numpy.get(),
																		 "floor") &&
			RegisterUFunc<UnaryUFunc<posit8_2, posit8_2, ufuncs::Ceil>>(numpy.get(),
																		"ceil") &&
			RegisterUFunc<UnaryUFunc<posit8_2, posit8_2, ufuncs::Trunc>>(numpy.get(),
																		 "trunc") &&
			RegisterUFunc<BinaryUFunc<posit8_2, posit8_2, ufuncs::NextAfter>>(
				numpy.get(), "nextafter");// &&

			//RegisterUFunc<UnaryUFunc<posit8_2, posit8_2, ufuncs::ToBinary<8UL>>>(numpy.get(), "binary_rep");

		return ok;
	}

	bool RegisterNumpyPosit8_2()
	{
		if (npy_posit8_2 != NPY_NOTYPE)
		{
			// Already initialized.
			return true;
		}
		if (!Initialize())
		{
			if (!PyErr_Occurred())
			{
				PyErr_SetString(PyExc_RuntimeError, "cannot load posit8_2 module.");
			}
			PyErr_Print();
			return false;
		}
		return true;
	}

	PyObject *Posit8_2Dtype()
	{
		return reinterpret_cast<PyObject *>(posit8_2_type_ptr);
	}

	int Posit8_2NumpyType() { return npy_posit8_2; }

	static PyMethodDef Posit8_2ModuleMethods[] = {
		{NULL, NULL, 0, NULL}
	};

	static struct PyModuleDef Posit8_2Module = {
		PyModuleDef_HEAD_INIT,
		"numpy_posit8_2",
		NULL,
		-1,
		Posit8_2ModuleMethods,
		NULL,
		NULL,
		NULL,
		NULL
	};

	PyMODINIT_FUNC
	PyInit_posit8_2(void)
	{
		PyObject *m;
		m = PyModule_Create(&Posit8_2Module);
		if (m == NULL)
			return NULL;
		RegisterNumpyPosit8_2();
		Py_INCREF(&posit8_2_type);
		Py_XINCREF(&NPyPosit8_2_Descr);
		if (PyModule_AddObject(m, "posit8_2", (PyObject *)&posit8_2_type) < 0)
		{
			Py_DECREF(&posit8_2_type);
			Py_DECREF(m);
			return NULL;
		}

		return m;
	}
} // namespace ceremorphic

хф
™э
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetypeИ
Њ
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring И
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshapeИ"serve*2.2.02v2.2.0-rc4-8-g2b96f3662b8рЪ

conv1d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:Ў* 
shared_nameconv1d_1/kernel
x
#conv1d_1/kernel/Read/ReadVariableOpReadVariableOpconv1d_1/kernel*#
_output_shapes
:Ў*
dtype0
r
conv1d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv1d_1/bias
k
!conv1d_1/bias/Read/ReadVariableOpReadVariableOpconv1d_1/bias*
_output_shapes
:*
dtype0
y
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	и*
shared_namedense_1/kernel
r
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes
:	и*
dtype0
p
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_1/bias
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes
:*
dtype0
d
SGD/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name
SGD/iter
]
SGD/iter/Read/ReadVariableOpReadVariableOpSGD/iter*
_output_shapes
: *
dtype0	
f
	SGD/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	SGD/decay
_
SGD/decay/Read/ReadVariableOpReadVariableOp	SGD/decay*
_output_shapes
: *
dtype0
v
SGD/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_nameSGD/learning_rate
o
%SGD/learning_rate/Read/ReadVariableOpReadVariableOpSGD/learning_rate*
_output_shapes
: *
dtype0
l
SGD/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameSGD/momentum
e
 SGD/momentum/Read/ReadVariableOpReadVariableOpSGD/momentum*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0

NoOpNoOp
п
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*™
value†BЭ BЦ
Н
layer-0
layer_with_weights-0
layer-1
layer-2
layer-3
layer-4
layer-5
layer_with_weights-1
layer-6
layer-7
		optimizer

trainable_variables
regularization_losses
	variables
	keras_api

signatures
 
h

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
R
trainable_variables
regularization_losses
	variables
	keras_api
R
trainable_variables
regularization_losses
	variables
	keras_api
R
trainable_variables
regularization_losses
	variables
 	keras_api
R
!trainable_variables
"regularization_losses
#	variables
$	keras_api
h

%kernel
&bias
'trainable_variables
(regularization_losses
)	variables
*	keras_api
R
+trainable_variables
,regularization_losses
-	variables
.	keras_api
6
/iter
	0decay
1learning_rate
2momentum

0
1
%2
&3
 

0
1
%2
&3
≠
3layer_metrics
4metrics
5non_trainable_variables

trainable_variables

6layers
regularization_losses
7layer_regularization_losses
	variables
 
[Y
VARIABLE_VALUEconv1d_1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv1d_1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
≠
8layer_metrics
9metrics
:non_trainable_variables
trainable_variables

;layers
regularization_losses
<layer_regularization_losses
	variables
 
 
 
≠
=layer_metrics
>metrics
?non_trainable_variables
trainable_variables

@layers
regularization_losses
Alayer_regularization_losses
	variables
 
 
 
≠
Blayer_metrics
Cmetrics
Dnon_trainable_variables
trainable_variables

Elayers
regularization_losses
Flayer_regularization_losses
	variables
 
 
 
≠
Glayer_metrics
Hmetrics
Inon_trainable_variables
trainable_variables

Jlayers
regularization_losses
Klayer_regularization_losses
	variables
 
 
 
≠
Llayer_metrics
Mmetrics
Nnon_trainable_variables
!trainable_variables

Olayers
"regularization_losses
Player_regularization_losses
#	variables
ZX
VARIABLE_VALUEdense_1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

%0
&1
 

%0
&1
≠
Qlayer_metrics
Rmetrics
Snon_trainable_variables
'trainable_variables

Tlayers
(regularization_losses
Ulayer_regularization_losses
)	variables
 
 
 
≠
Vlayer_metrics
Wmetrics
Xnon_trainable_variables
+trainable_variables

Ylayers
,regularization_losses
Zlayer_regularization_losses
-	variables
GE
VARIABLE_VALUESGD/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUE	SGD/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUESGD/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUESGD/momentum-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
 

[0
 
8
0
1
2
3
4
5
6
7
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
4
	\total
	]count
^	variables
_	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

\0
]1

^	variables
Д
serving_default_input_2Placeholder*,
_output_shapes
:€€€€€€€€€Ў*
dtype0*!
shape:€€€€€€€€€Ў
„
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_2conv1d_1/kernelconv1d_1/biasdense_1/kerneldense_1/bias*
Tin	
2*
Tout
2*'
_output_shapes
:€€€€€€€€€*&
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*.
f)R'
%__inference_signature_wrapper_2894413
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
ќ
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#conv1d_1/kernel/Read/ReadVariableOp!conv1d_1/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOpSGD/iter/Read/ReadVariableOpSGD/decay/Read/ReadVariableOp%SGD/learning_rate/Read/ReadVariableOp SGD/momentum/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOpConst*
Tin
2	*
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*)
f$R"
 __inference__traced_save_2894685
Б
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv1d_1/kernelconv1d_1/biasdense_1/kerneldense_1/biasSGD/iter	SGD/decaySGD/learning_rateSGD/momentumtotalcount*
Tin
2*
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*,
f'R%
#__inference__traced_restore_2894727ша
Н'
Ј
Y__inference_simple_convolutional_network_layer_call_and_return_conditional_losses_2894333

inputs
conv1d_1_2894309
conv1d_1_2894311
dense_1_2894318
dense_1_2894320
identityИҐ conv1d_1/StatefulPartitionedCallҐdense_1/StatefulPartitionedCallҐ!dropout_1/StatefulPartitionedCallщ
 conv1d_1/StatefulPartitionedCallStatefulPartitionedCallinputsconv1d_1_2894309conv1d_1_2894311*
Tin
2*
Tout
2*+
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*N
fIRG
E__inference_conv1d_1_layer_call_and_return_conditional_losses_28941282"
 conv1d_1/StatefulPartitionedCallж
activation_2/PartitionedCallPartitionedCall)conv1d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*+
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*R
fMRK
I__inference_activation_2_layer_call_and_return_conditional_losses_28941672
activation_2/PartitionedCallл
max_pooling1d_1/PartitionedCallPartitionedCall%activation_2/PartitionedCall:output:0*
Tin
2*
Tout
2*+
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*U
fPRN
L__inference_max_pooling1d_1_layer_call_and_return_conditional_losses_28941472!
max_pooling1d_1/PartitionedCallф
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall(max_pooling1d_1/PartitionedCall:output:0*
Tin
2*
Tout
2*+
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*O
fJRH
F__inference_dropout_1_layer_call_and_return_conditional_losses_28941882#
!dropout_1/StatefulPartitionedCallџ
flatten_1/PartitionedCallPartitionedCall*dropout_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:€€€€€€€€€и* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*O
fJRH
F__inference_flatten_1_layer_call_and_return_conditional_losses_28942122
flatten_1/PartitionedCallМ
dense_1/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_1_2894318dense_1_2894320*
Tin
2*
Tout
2*'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_28942382!
dense_1/StatefulPartitionedCallб
activation_3/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*R
fMRK
I__inference_activation_3_layer_call_and_return_conditional_losses_28942592
activation_3/PartitionedCallµ
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_1_2894318*
_output_shapes
:	и*
dtype022
0dense_1/kernel/Regularizer/Square/ReadVariableOpі
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	и2#
!dense_1/kernel/Regularizer/SquareХ
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_1/kernel/Regularizer/ConstЇ
dense_1/kernel/Regularizer/SumSum%dense_1/kernel/Regularizer/Square:y:0)dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/SumЙ
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2"
 dense_1/kernel/Regularizer/mul/xЉ
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/mulЙ
 dense_1/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_1/kernel/Regularizer/add/xє
dense_1/kernel/Regularizer/addAddV2)dense_1/kernel/Regularizer/add/x:output:0"dense_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/addв
IdentityIdentity%activation_3/PartitionedCall:output:0!^conv1d_1/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:€€€€€€€€€Ў::::2D
 conv1d_1/StatefulPartitionedCall conv1d_1/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall:T P
,
_output_shapes
:€€€€€€€€€Ў
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
І
±
>__inference_simple_convolutional_network_layer_call_fn_2894522

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCallБ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*'
_output_shapes
:€€€€€€€€€*&
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*b
f]R[
Y__inference_simple_convolutional_network_layer_call_and_return_conditional_losses_28943732
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:€€€€€€€€€Ў::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:€€€€€€€€€Ў
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ѓ
e
F__inference_dropout_1_layer_call_and_return_conditional_losses_2894544

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *яќЯ?2
dropout/Constw
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:€€€€€€€€€2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shapeƒ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:€€€€€€€€€*
dtype0*

seed2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ш–K>2
dropout/GreaterEqual/y¬
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:€€€€€€€€€2
dropout/GreaterEqualГ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:€€€€€€€€€2
dropout/Cast~
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:€€€€€€€€€2
dropout/Mul_1i
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0**
_input_shapes
:€€€€€€€€€:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
о6
є
Y__inference_simple_convolutional_network_layer_call_and_return_conditional_losses_2894458

inputs8
4conv1d_1_conv1d_expanddims_1_readvariableop_resource,
(conv1d_1_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource
identityИВ
conv1d_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2 
conv1d_1/conv1d/ExpandDims/dim≤
conv1d_1/conv1d/ExpandDims
ExpandDimsinputs'conv1d_1/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:€€€€€€€€€Ў2
conv1d_1/conv1d/ExpandDims‘
+conv1d_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_1_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:Ў*
dtype02-
+conv1d_1/conv1d/ExpandDims_1/ReadVariableOpЖ
 conv1d_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_1/conv1d/ExpandDims_1/dim№
conv1d_1/conv1d/ExpandDims_1
ExpandDims3conv1d_1/conv1d/ExpandDims_1/ReadVariableOp:value:0)conv1d_1/conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:Ў2
conv1d_1/conv1d/ExpandDims_1џ
conv1d_1/conv1dConv2D#conv1d_1/conv1d/ExpandDims:output:0%conv1d_1/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:€€€€€€€€€*
paddingVALID*
strides
2
conv1d_1/conv1d§
conv1d_1/conv1d/SqueezeSqueezeconv1d_1/conv1d:output:0*
T0*+
_output_shapes
:€€€€€€€€€*
squeeze_dims
2
conv1d_1/conv1d/SqueezeІ
conv1d_1/BiasAdd/ReadVariableOpReadVariableOp(conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv1d_1/BiasAdd/ReadVariableOp∞
conv1d_1/BiasAddBiasAdd conv1d_1/conv1d/Squeeze:output:0'conv1d_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€2
conv1d_1/BiasAdd
activation_2/ReluReluconv1d_1/BiasAdd:output:0*
T0*+
_output_shapes
:€€€€€€€€€2
activation_2/ReluВ
max_pooling1d_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2 
max_pooling1d_1/ExpandDims/dim 
max_pooling1d_1/ExpandDims
ExpandDimsactivation_2/Relu:activations:0'max_pooling1d_1/ExpandDims/dim:output:0*
T0*/
_output_shapes
:€€€€€€€€€2
max_pooling1d_1/ExpandDimsѕ
max_pooling1d_1/MaxPoolMaxPool#max_pooling1d_1/ExpandDims:output:0*/
_output_shapes
:€€€€€€€€€*
ksize
*
paddingVALID*
strides
2
max_pooling1d_1/MaxPoolђ
max_pooling1d_1/SqueezeSqueeze max_pooling1d_1/MaxPool:output:0*
T0*+
_output_shapes
:€€€€€€€€€*
squeeze_dims
2
max_pooling1d_1/Squeezew
dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *яќЯ?2
dropout_1/dropout/Constѓ
dropout_1/dropout/MulMul max_pooling1d_1/Squeeze:output:0 dropout_1/dropout/Const:output:0*
T0*+
_output_shapes
:€€€€€€€€€2
dropout_1/dropout/MulВ
dropout_1/dropout/ShapeShape max_pooling1d_1/Squeeze:output:0*
T0*
_output_shapes
:2
dropout_1/dropout/Shapeв
.dropout_1/dropout/random_uniform/RandomUniformRandomUniform dropout_1/dropout/Shape:output:0*
T0*+
_output_shapes
:€€€€€€€€€*
dtype0*

seed20
.dropout_1/dropout/random_uniform/RandomUniformЙ
 dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ш–K>2"
 dropout_1/dropout/GreaterEqual/yк
dropout_1/dropout/GreaterEqualGreaterEqual7dropout_1/dropout/random_uniform/RandomUniform:output:0)dropout_1/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:€€€€€€€€€2 
dropout_1/dropout/GreaterEqual°
dropout_1/dropout/CastCast"dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:€€€€€€€€€2
dropout_1/dropout/Cast¶
dropout_1/dropout/Mul_1Muldropout_1/dropout/Mul:z:0dropout_1/dropout/Cast:y:0*
T0*+
_output_shapes
:€€€€€€€€€2
dropout_1/dropout/Mul_1s
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€h  2
flatten_1/ConstЫ
flatten_1/ReshapeReshapedropout_1/dropout/Mul_1:z:0flatten_1/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€и2
flatten_1/Reshape¶
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	и*
dtype02
dense_1/MatMul/ReadVariableOpЯ
dense_1/MatMulMatMulflatten_1/Reshape:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
dense_1/MatMul§
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_1/BiasAdd/ReadVariableOp°
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
dense_1/BiasAddz
activation_3/TanhTanhdense_1/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€2
activation_3/Tanhћ
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	и*
dtype022
0dense_1/kernel/Regularizer/Square/ReadVariableOpі
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	и2#
!dense_1/kernel/Regularizer/SquareХ
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_1/kernel/Regularizer/ConstЇ
dense_1/kernel/Regularizer/SumSum%dense_1/kernel/Regularizer/Square:y:0)dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/SumЙ
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2"
 dense_1/kernel/Regularizer/mul/xЉ
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/mulЙ
 dense_1/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_1/kernel/Regularizer/add/xє
dense_1/kernel/Regularizer/addAddV2)dense_1/kernel/Regularizer/add/x:output:0"dense_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/addi
IdentityIdentityactivation_3/Tanh:y:0*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:€€€€€€€€€Ў:::::T P
,
_output_shapes
:€€€€€€€€€Ў
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Ё%
У
Y__inference_simple_convolutional_network_layer_call_and_return_conditional_losses_2894373

inputs
conv1d_1_2894349
conv1d_1_2894351
dense_1_2894358
dense_1_2894360
identityИҐ conv1d_1/StatefulPartitionedCallҐdense_1/StatefulPartitionedCallщ
 conv1d_1/StatefulPartitionedCallStatefulPartitionedCallinputsconv1d_1_2894349conv1d_1_2894351*
Tin
2*
Tout
2*+
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*N
fIRG
E__inference_conv1d_1_layer_call_and_return_conditional_losses_28941282"
 conv1d_1/StatefulPartitionedCallж
activation_2/PartitionedCallPartitionedCall)conv1d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*+
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*R
fMRK
I__inference_activation_2_layer_call_and_return_conditional_losses_28941672
activation_2/PartitionedCallл
max_pooling1d_1/PartitionedCallPartitionedCall%activation_2/PartitionedCall:output:0*
Tin
2*
Tout
2*+
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*U
fPRN
L__inference_max_pooling1d_1_layer_call_and_return_conditional_losses_28941472!
max_pooling1d_1/PartitionedCall№
dropout_1/PartitionedCallPartitionedCall(max_pooling1d_1/PartitionedCall:output:0*
Tin
2*
Tout
2*+
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*O
fJRH
F__inference_dropout_1_layer_call_and_return_conditional_losses_28941932
dropout_1/PartitionedCall”
flatten_1/PartitionedCallPartitionedCall"dropout_1/PartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:€€€€€€€€€и* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*O
fJRH
F__inference_flatten_1_layer_call_and_return_conditional_losses_28942122
flatten_1/PartitionedCallМ
dense_1/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_1_2894358dense_1_2894360*
Tin
2*
Tout
2*'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_28942382!
dense_1/StatefulPartitionedCallб
activation_3/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*R
fMRK
I__inference_activation_3_layer_call_and_return_conditional_losses_28942592
activation_3/PartitionedCallµ
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_1_2894358*
_output_shapes
:	и*
dtype022
0dense_1/kernel/Regularizer/Square/ReadVariableOpі
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	и2#
!dense_1/kernel/Regularizer/SquareХ
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_1/kernel/Regularizer/ConstЇ
dense_1/kernel/Regularizer/SumSum%dense_1/kernel/Regularizer/Square:y:0)dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/SumЙ
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2"
 dense_1/kernel/Regularizer/mul/xЉ
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/mulЙ
 dense_1/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_1/kernel/Regularizer/add/xє
dense_1/kernel/Regularizer/addAddV2)dense_1/kernel/Regularizer/add/x:output:0"dense_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/addЊ
IdentityIdentity%activation_3/PartitionedCall:output:0!^conv1d_1/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:€€€€€€€€€Ў::::2D
 conv1d_1/StatefulPartitionedCall conv1d_1/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:T P
,
_output_shapes
:€€€€€€€€€Ў
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
™
≤
>__inference_simple_convolutional_network_layer_call_fn_2894384
input_2
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCallВ
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*'
_output_shapes
:€€€€€€€€€*&
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*b
f]R[
Y__inference_simple_convolutional_network_layer_call_and_return_conditional_losses_28943732
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:€€€€€€€€€Ў::::22
StatefulPartitionedCallStatefulPartitionedCall:U Q
,
_output_shapes
:€€€€€€€€€Ў
!
_user_specified_name	input_2:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
т
ђ
D__inference_dense_1_layer_call_and_return_conditional_losses_2894238

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	и*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2	
BiasAddƒ
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	и*
dtype022
0dense_1/kernel/Regularizer/Square/ReadVariableOpі
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	и2#
!dense_1/kernel/Regularizer/SquareХ
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_1/kernel/Regularizer/ConstЇ
dense_1/kernel/Regularizer/SumSum%dense_1/kernel/Regularizer/Square:y:0)dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/SumЙ
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2"
 dense_1/kernel/Regularizer/mul/xЉ
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/mulЙ
 dense_1/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_1/kernel/Regularizer/add/xє
dense_1/kernel/Regularizer/addAddV2)dense_1/kernel/Regularizer/add/x:output:0"dense_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/addd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*/
_input_shapes
:€€€€€€€€€и:::P L
(
_output_shapes
:€€€€€€€€€и
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
ѓ
e
F__inference_dropout_1_layer_call_and_return_conditional_losses_2894188

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *яќЯ?2
dropout/Constw
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:€€€€€€€€€2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shapeƒ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:€€€€€€€€€*
dtype0*

seed2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ш–K>2
dropout/GreaterEqual/y¬
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:€€€€€€€€€2
dropout/GreaterEqualГ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:€€€€€€€€€2
dropout/Cast~
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:€€€€€€€€€2
dropout/Mul_1i
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0**
_input_shapes
:€€€€€€€€€:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
«
e
I__inference_activation_2_layer_call_and_return_conditional_losses_2894527

inputs
identityR
ReluReluinputs*
T0*+
_output_shapes
:€€€€€€€€€2
Reluj
IdentityIdentityRelu:activations:0*
T0*+
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0**
_input_shapes
:€€€€€€€€€:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
Ѓ

*__inference_conv1d_1_layer_call_fn_2894138

inputs
unknown
	unknown_0
identityИҐStatefulPartitionedCallа
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*N
fIRG
E__inference_conv1d_1_layer_call_and_return_conditional_losses_28941282
StatefulPartitionedCallЫ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*<
_input_shapes+
):€€€€€€€€€€€€€€€€€€Ў::22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€Ў
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
ґ
b
F__inference_flatten_1_layer_call_and_return_conditional_losses_2894212

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€h  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:€€€€€€€€€и2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:€€€€€€€€€и2

Identity"
identityIdentity:output:0**
_input_shapes
:€€€€€€€€€:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
ў
d
F__inference_dropout_1_layer_call_and_return_conditional_losses_2894193

inputs

identity_1^
IdentityIdentityinputs*
T0*+
_output_shapes
:€€€€€€€€€2

Identitym

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:€€€€€€€€€2

Identity_1"!

identity_1Identity_1:output:0**
_input_shapes
:€€€€€€€€€:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
ю
G
+__inference_flatten_1_layer_call_fn_2894570

inputs
identity£
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*(
_output_shapes
:€€€€€€€€€и* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*O
fJRH
F__inference_flatten_1_layer_call_and_return_conditional_losses_28942122
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:€€€€€€€€€и2

Identity"
identityIdentity:output:0**
_input_shapes
:€€€€€€€€€:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
Ґ-
є
Y__inference_simple_convolutional_network_layer_call_and_return_conditional_losses_2894496

inputs8
4conv1d_1_conv1d_expanddims_1_readvariableop_resource,
(conv1d_1_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource
identityИВ
conv1d_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2 
conv1d_1/conv1d/ExpandDims/dim≤
conv1d_1/conv1d/ExpandDims
ExpandDimsinputs'conv1d_1/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:€€€€€€€€€Ў2
conv1d_1/conv1d/ExpandDims‘
+conv1d_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_1_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:Ў*
dtype02-
+conv1d_1/conv1d/ExpandDims_1/ReadVariableOpЖ
 conv1d_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_1/conv1d/ExpandDims_1/dim№
conv1d_1/conv1d/ExpandDims_1
ExpandDims3conv1d_1/conv1d/ExpandDims_1/ReadVariableOp:value:0)conv1d_1/conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:Ў2
conv1d_1/conv1d/ExpandDims_1џ
conv1d_1/conv1dConv2D#conv1d_1/conv1d/ExpandDims:output:0%conv1d_1/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:€€€€€€€€€*
paddingVALID*
strides
2
conv1d_1/conv1d§
conv1d_1/conv1d/SqueezeSqueezeconv1d_1/conv1d:output:0*
T0*+
_output_shapes
:€€€€€€€€€*
squeeze_dims
2
conv1d_1/conv1d/SqueezeІ
conv1d_1/BiasAdd/ReadVariableOpReadVariableOp(conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv1d_1/BiasAdd/ReadVariableOp∞
conv1d_1/BiasAddBiasAdd conv1d_1/conv1d/Squeeze:output:0'conv1d_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€2
conv1d_1/BiasAdd
activation_2/ReluReluconv1d_1/BiasAdd:output:0*
T0*+
_output_shapes
:€€€€€€€€€2
activation_2/ReluВ
max_pooling1d_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2 
max_pooling1d_1/ExpandDims/dim 
max_pooling1d_1/ExpandDims
ExpandDimsactivation_2/Relu:activations:0'max_pooling1d_1/ExpandDims/dim:output:0*
T0*/
_output_shapes
:€€€€€€€€€2
max_pooling1d_1/ExpandDimsѕ
max_pooling1d_1/MaxPoolMaxPool#max_pooling1d_1/ExpandDims:output:0*/
_output_shapes
:€€€€€€€€€*
ksize
*
paddingVALID*
strides
2
max_pooling1d_1/MaxPoolђ
max_pooling1d_1/SqueezeSqueeze max_pooling1d_1/MaxPool:output:0*
T0*+
_output_shapes
:€€€€€€€€€*
squeeze_dims
2
max_pooling1d_1/SqueezeМ
dropout_1/IdentityIdentity max_pooling1d_1/Squeeze:output:0*
T0*+
_output_shapes
:€€€€€€€€€2
dropout_1/Identitys
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€h  2
flatten_1/ConstЫ
flatten_1/ReshapeReshapedropout_1/Identity:output:0flatten_1/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€и2
flatten_1/Reshape¶
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	и*
dtype02
dense_1/MatMul/ReadVariableOpЯ
dense_1/MatMulMatMulflatten_1/Reshape:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
dense_1/MatMul§
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_1/BiasAdd/ReadVariableOp°
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
dense_1/BiasAddz
activation_3/TanhTanhdense_1/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€2
activation_3/Tanhћ
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	и*
dtype022
0dense_1/kernel/Regularizer/Square/ReadVariableOpі
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	и2#
!dense_1/kernel/Regularizer/SquareХ
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_1/kernel/Regularizer/ConstЇ
dense_1/kernel/Regularizer/SumSum%dense_1/kernel/Regularizer/Square:y:0)dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/SumЙ
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2"
 dense_1/kernel/Regularizer/mul/xЉ
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/mulЙ
 dense_1/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_1/kernel/Regularizer/add/xє
dense_1/kernel/Regularizer/addAddV2)dense_1/kernel/Regularizer/add/x:output:0"dense_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/addi
IdentityIdentityactivation_3/Tanh:y:0*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:€€€€€€€€€Ў:::::T P
,
_output_shapes
:€€€€€€€€€Ў
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
™1
Ъ
#__inference__traced_restore_2894727
file_prefix$
 assignvariableop_conv1d_1_kernel$
 assignvariableop_1_conv1d_1_bias%
!assignvariableop_2_dense_1_kernel#
assignvariableop_3_dense_1_bias
assignvariableop_4_sgd_iter 
assignvariableop_5_sgd_decay(
$assignvariableop_6_sgd_learning_rate#
assignvariableop_7_sgd_momentum
assignvariableop_8_total
assignvariableop_9_count
identity_11ИҐAssignVariableOpҐAssignVariableOp_1ҐAssignVariableOp_2ҐAssignVariableOp_3ҐAssignVariableOp_4ҐAssignVariableOp_5ҐAssignVariableOp_6ҐAssignVariableOp_7ҐAssignVariableOp_8ҐAssignVariableOp_9Ґ	RestoreV2ҐRestoreV2_1Л
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:
*
dtype0*Ч
valueНBК
B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE2
RestoreV2/tensor_namesҐ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:
*
dtype0*'
valueB
B B B B B B B B B B 2
RestoreV2/shape_and_slicesЁ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*<
_output_shapes*
(::::::::::*
dtypes
2
	2
	RestoreV2X
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:2

IdentityР
AssignVariableOpAssignVariableOp assignvariableop_conv1d_1_kernelIdentity:output:0*
_output_shapes
 *
dtype02
AssignVariableOp\

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:2

Identity_1Ц
AssignVariableOp_1AssignVariableOp assignvariableop_1_conv1d_1_biasIdentity_1:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_1\

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:2

Identity_2Ч
AssignVariableOp_2AssignVariableOp!assignvariableop_2_dense_1_kernelIdentity_2:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_2\

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:2

Identity_3Х
AssignVariableOp_3AssignVariableOpassignvariableop_3_dense_1_biasIdentity_3:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_3\

Identity_4IdentityRestoreV2:tensors:4*
T0	*
_output_shapes
:2

Identity_4С
AssignVariableOp_4AssignVariableOpassignvariableop_4_sgd_iterIdentity_4:output:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_4\

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:2

Identity_5Т
AssignVariableOp_5AssignVariableOpassignvariableop_5_sgd_decayIdentity_5:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_5\

Identity_6IdentityRestoreV2:tensors:6*
T0*
_output_shapes
:2

Identity_6Ъ
AssignVariableOp_6AssignVariableOp$assignvariableop_6_sgd_learning_rateIdentity_6:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_6\

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:2

Identity_7Х
AssignVariableOp_7AssignVariableOpassignvariableop_7_sgd_momentumIdentity_7:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_7\

Identity_8IdentityRestoreV2:tensors:8*
T0*
_output_shapes
:2

Identity_8О
AssignVariableOp_8AssignVariableOpassignvariableop_8_totalIdentity_8:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_8\

Identity_9IdentityRestoreV2:tensors:9*
T0*
_output_shapes
:2

Identity_9О
AssignVariableOp_9AssignVariableOpassignvariableop_9_countIdentity_9:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_9®
RestoreV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2_1/tensor_namesФ
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
RestoreV2_1/shape_and_slicesƒ
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
_output_shapes
:*
dtypes
22
RestoreV2_19
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpЇ
Identity_10Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_10«
Identity_11IdentityIdentity_10:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: 2
Identity_11"#
identity_11Identity_11:output:0*=
_input_shapes,
*: ::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92
	RestoreV2	RestoreV22
RestoreV2_1RestoreV2_1:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: 
т
ђ
D__inference_dense_1_layer_call_and_return_conditional_losses_2894596

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	и*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2	
BiasAddƒ
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	и*
dtype022
0dense_1/kernel/Regularizer/Square/ReadVariableOpі
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	и2#
!dense_1/kernel/Regularizer/SquareХ
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_1/kernel/Regularizer/ConstЇ
dense_1/kernel/Regularizer/SumSum%dense_1/kernel/Regularizer/Square:y:0)dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/SumЙ
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2"
 dense_1/kernel/Regularizer/mul/xЉ
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/mulЙ
 dense_1/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_1/kernel/Regularizer/add/xє
dense_1/kernel/Regularizer/addAddV2)dense_1/kernel/Regularizer/add/x:output:0"dense_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/addd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*/
_input_shapes
:€€€€€€€€€и:::P L
(
_output_shapes
:€€€€€€€€€и
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
І
±
>__inference_simple_convolutional_network_layer_call_fn_2894509

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCallБ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*'
_output_shapes
:€€€€€€€€€*&
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*b
f]R[
Y__inference_simple_convolutional_network_layer_call_and_return_conditional_losses_28943332
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:€€€€€€€€€Ў::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:€€€€€€€€€Ў
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ґ
b
F__inference_flatten_1_layer_call_and_return_conditional_losses_2894565

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€h  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:€€€€€€€€€и2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:€€€€€€€€€и2

Identity"
identityIdentity:output:0**
_input_shapes
:€€€€€€€€€:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
≠
e
I__inference_activation_3_layer_call_and_return_conditional_losses_2894259

inputs
identityN
TanhTanhinputs*
T0*'
_output_shapes
:€€€€€€€€€2
Tanh\
IdentityIdentityTanh:y:0*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*&
_input_shapes
:€€€€€€€€€:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
ў
M
1__inference_max_pooling1d_1_layer_call_fn_2894153

inputs
identityЊ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*U
fPRN
L__inference_max_pooling1d_1_layer_call_and_return_conditional_losses_28941472
PartitionedCallВ
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€:e a
=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
ѓ
o
__inference_loss_fn_0_2894628=
9dense_1_kernel_regularizer_square_readvariableop_resource
identityИя
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp9dense_1_kernel_regularizer_square_readvariableop_resource*
_output_shapes
:	и*
dtype022
0dense_1/kernel/Regularizer/Square/ReadVariableOpі
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	и2#
!dense_1/kernel/Regularizer/SquareХ
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_1/kernel/Regularizer/ConstЇ
dense_1/kernel/Regularizer/SumSum%dense_1/kernel/Regularizer/Square:y:0)dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/SumЙ
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2"
 dense_1/kernel/Regularizer/mul/xЉ
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/mulЙ
 dense_1/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_1/kernel/Regularizer/add/xє
dense_1/kernel/Regularizer/addAddV2)dense_1/kernel/Regularizer/add/x:output:0"dense_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/adde
IdentityIdentity"dense_1/kernel/Regularizer/add:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:: 

_output_shapes
: 
ж2
ч
"__inference__wrapped_model_2894112
input_2U
Qsimple_convolutional_network_conv1d_1_conv1d_expanddims_1_readvariableop_resourceI
Esimple_convolutional_network_conv1d_1_biasadd_readvariableop_resourceG
Csimple_convolutional_network_dense_1_matmul_readvariableop_resourceH
Dsimple_convolutional_network_dense_1_biasadd_readvariableop_resource
identityИЉ
;simple_convolutional_network/conv1d_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2=
;simple_convolutional_network/conv1d_1/conv1d/ExpandDims/dimК
7simple_convolutional_network/conv1d_1/conv1d/ExpandDims
ExpandDimsinput_2Dsimple_convolutional_network/conv1d_1/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:€€€€€€€€€Ў29
7simple_convolutional_network/conv1d_1/conv1d/ExpandDimsЂ
Hsimple_convolutional_network/conv1d_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpQsimple_convolutional_network_conv1d_1_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:Ў*
dtype02J
Hsimple_convolutional_network/conv1d_1/conv1d/ExpandDims_1/ReadVariableOpј
=simple_convolutional_network/conv1d_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2?
=simple_convolutional_network/conv1d_1/conv1d/ExpandDims_1/dim–
9simple_convolutional_network/conv1d_1/conv1d/ExpandDims_1
ExpandDimsPsimple_convolutional_network/conv1d_1/conv1d/ExpandDims_1/ReadVariableOp:value:0Fsimple_convolutional_network/conv1d_1/conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:Ў2;
9simple_convolutional_network/conv1d_1/conv1d/ExpandDims_1ѕ
,simple_convolutional_network/conv1d_1/conv1dConv2D@simple_convolutional_network/conv1d_1/conv1d/ExpandDims:output:0Bsimple_convolutional_network/conv1d_1/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:€€€€€€€€€*
paddingVALID*
strides
2.
,simple_convolutional_network/conv1d_1/conv1dы
4simple_convolutional_network/conv1d_1/conv1d/SqueezeSqueeze5simple_convolutional_network/conv1d_1/conv1d:output:0*
T0*+
_output_shapes
:€€€€€€€€€*
squeeze_dims
26
4simple_convolutional_network/conv1d_1/conv1d/Squeezeю
<simple_convolutional_network/conv1d_1/BiasAdd/ReadVariableOpReadVariableOpEsimple_convolutional_network_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02>
<simple_convolutional_network/conv1d_1/BiasAdd/ReadVariableOp§
-simple_convolutional_network/conv1d_1/BiasAddBiasAdd=simple_convolutional_network/conv1d_1/conv1d/Squeeze:output:0Dsimple_convolutional_network/conv1d_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€2/
-simple_convolutional_network/conv1d_1/BiasAdd÷
.simple_convolutional_network/activation_2/ReluRelu6simple_convolutional_network/conv1d_1/BiasAdd:output:0*
T0*+
_output_shapes
:€€€€€€€€€20
.simple_convolutional_network/activation_2/ReluЉ
;simple_convolutional_network/max_pooling1d_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2=
;simple_convolutional_network/max_pooling1d_1/ExpandDims/dimЊ
7simple_convolutional_network/max_pooling1d_1/ExpandDims
ExpandDims<simple_convolutional_network/activation_2/Relu:activations:0Dsimple_convolutional_network/max_pooling1d_1/ExpandDims/dim:output:0*
T0*/
_output_shapes
:€€€€€€€€€29
7simple_convolutional_network/max_pooling1d_1/ExpandDims¶
4simple_convolutional_network/max_pooling1d_1/MaxPoolMaxPool@simple_convolutional_network/max_pooling1d_1/ExpandDims:output:0*/
_output_shapes
:€€€€€€€€€*
ksize
*
paddingVALID*
strides
26
4simple_convolutional_network/max_pooling1d_1/MaxPoolГ
4simple_convolutional_network/max_pooling1d_1/SqueezeSqueeze=simple_convolutional_network/max_pooling1d_1/MaxPool:output:0*
T0*+
_output_shapes
:€€€€€€€€€*
squeeze_dims
26
4simple_convolutional_network/max_pooling1d_1/Squeezeг
/simple_convolutional_network/dropout_1/IdentityIdentity=simple_convolutional_network/max_pooling1d_1/Squeeze:output:0*
T0*+
_output_shapes
:€€€€€€€€€21
/simple_convolutional_network/dropout_1/Identity≠
,simple_convolutional_network/flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€h  2.
,simple_convolutional_network/flatten_1/ConstП
.simple_convolutional_network/flatten_1/ReshapeReshape8simple_convolutional_network/dropout_1/Identity:output:05simple_convolutional_network/flatten_1/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€и20
.simple_convolutional_network/flatten_1/Reshapeэ
:simple_convolutional_network/dense_1/MatMul/ReadVariableOpReadVariableOpCsimple_convolutional_network_dense_1_matmul_readvariableop_resource*
_output_shapes
:	и*
dtype02<
:simple_convolutional_network/dense_1/MatMul/ReadVariableOpУ
+simple_convolutional_network/dense_1/MatMulMatMul7simple_convolutional_network/flatten_1/Reshape:output:0Bsimple_convolutional_network/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2-
+simple_convolutional_network/dense_1/MatMulы
;simple_convolutional_network/dense_1/BiasAdd/ReadVariableOpReadVariableOpDsimple_convolutional_network_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02=
;simple_convolutional_network/dense_1/BiasAdd/ReadVariableOpХ
,simple_convolutional_network/dense_1/BiasAddBiasAdd5simple_convolutional_network/dense_1/MatMul:product:0Csimple_convolutional_network/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2.
,simple_convolutional_network/dense_1/BiasAdd—
.simple_convolutional_network/activation_3/TanhTanh5simple_convolutional_network/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€20
.simple_convolutional_network/activation_3/TanhЖ
IdentityIdentity2simple_convolutional_network/activation_3/Tanh:y:0*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:€€€€€€€€€Ў:::::U Q
,
_output_shapes
:€€€€€€€€€Ў
!
_user_specified_name	input_2:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Д
G
+__inference_dropout_1_layer_call_fn_2894559

inputs
identity¶
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*+
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*O
fJRH
F__inference_dropout_1_layer_call_and_return_conditional_losses_28941932
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0**
_input_shapes
:€€€€€€€€€:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
й
h
L__inference_max_pooling1d_1_layer_call_and_return_conditional_losses_2894147

inputs
identityb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dimУ

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2

ExpandDims±
MaxPoolMaxPoolExpandDims:output:0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*
ksize
*
paddingVALID*
strides
2	
MaxPoolО
SqueezeSqueezeMaxPool:output:0*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€*
squeeze_dims
2	
Squeezez
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€:e a
=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
Џ
Щ
%__inference_signature_wrapper_2894413
input_2
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCallЋ
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*'
_output_shapes
:€€€€€€€€€*&
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*+
f&R$
"__inference__wrapped_model_28941122
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:€€€€€€€€€Ў::::22
StatefulPartitionedCallStatefulPartitionedCall:U Q
,
_output_shapes
:€€€€€€€€€Ў
!
_user_specified_name	input_2:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
™
≤
>__inference_simple_convolutional_network_layer_call_fn_2894344
input_2
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCallВ
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*'
_output_shapes
:€€€€€€€€€*&
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*b
f]R[
Y__inference_simple_convolutional_network_layer_call_and_return_conditional_losses_28943332
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:€€€€€€€€€Ў::::22
StatefulPartitionedCallStatefulPartitionedCall:U Q
,
_output_shapes
:€€€€€€€€€Ў
!
_user_specified_name	input_2:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
«
e
I__inference_activation_2_layer_call_and_return_conditional_losses_2894167

inputs
identityR
ReluReluinputs*
T0*+
_output_shapes
:€€€€€€€€€2
Reluj
IdentityIdentityRelu:activations:0*
T0*+
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0**
_input_shapes
:€€€€€€€€€:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
Р
d
+__inference_dropout_1_layer_call_fn_2894554

inputs
identityИҐStatefulPartitionedCallЊ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*+
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*O
fJRH
F__inference_dropout_1_layer_call_and_return_conditional_losses_28941882
StatefulPartitionedCallТ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0**
_input_shapes
:€€€€€€€€€22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
ў
d
F__inference_dropout_1_layer_call_and_return_conditional_losses_2894549

inputs

identity_1^
IdentityIdentityinputs*
T0*+
_output_shapes
:€€€€€€€€€2

Identitym

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:€€€€€€€€€2

Identity_1"!

identity_1Identity_1:output:0**
_input_shapes
:€€€€€€€€€:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
ш
~
)__inference_dense_1_layer_call_fn_2894605

inputs
unknown
	unknown_0
identityИҐStatefulPartitionedCall“
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_28942382
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*/
_input_shapes
:€€€€€€€€€и::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:€€€€€€€€€и
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
К
J
.__inference_activation_2_layer_call_fn_2894532

inputs
identity©
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*+
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*R
fMRK
I__inference_activation_2_layer_call_and_return_conditional_losses_28941672
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0**
_input_shapes
:€€€€€€€€€:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
а%
Ф
Y__inference_simple_convolutional_network_layer_call_and_return_conditional_losses_2894303
input_2
conv1d_1_2894279
conv1d_1_2894281
dense_1_2894288
dense_1_2894290
identityИҐ conv1d_1/StatefulPartitionedCallҐdense_1/StatefulPartitionedCallъ
 conv1d_1/StatefulPartitionedCallStatefulPartitionedCallinput_2conv1d_1_2894279conv1d_1_2894281*
Tin
2*
Tout
2*+
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*N
fIRG
E__inference_conv1d_1_layer_call_and_return_conditional_losses_28941282"
 conv1d_1/StatefulPartitionedCallж
activation_2/PartitionedCallPartitionedCall)conv1d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*+
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*R
fMRK
I__inference_activation_2_layer_call_and_return_conditional_losses_28941672
activation_2/PartitionedCallл
max_pooling1d_1/PartitionedCallPartitionedCall%activation_2/PartitionedCall:output:0*
Tin
2*
Tout
2*+
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*U
fPRN
L__inference_max_pooling1d_1_layer_call_and_return_conditional_losses_28941472!
max_pooling1d_1/PartitionedCall№
dropout_1/PartitionedCallPartitionedCall(max_pooling1d_1/PartitionedCall:output:0*
Tin
2*
Tout
2*+
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*O
fJRH
F__inference_dropout_1_layer_call_and_return_conditional_losses_28941932
dropout_1/PartitionedCall”
flatten_1/PartitionedCallPartitionedCall"dropout_1/PartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:€€€€€€€€€и* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*O
fJRH
F__inference_flatten_1_layer_call_and_return_conditional_losses_28942122
flatten_1/PartitionedCallМ
dense_1/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_1_2894288dense_1_2894290*
Tin
2*
Tout
2*'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_28942382!
dense_1/StatefulPartitionedCallб
activation_3/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*R
fMRK
I__inference_activation_3_layer_call_and_return_conditional_losses_28942592
activation_3/PartitionedCallµ
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_1_2894288*
_output_shapes
:	и*
dtype022
0dense_1/kernel/Regularizer/Square/ReadVariableOpі
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	и2#
!dense_1/kernel/Regularizer/SquareХ
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_1/kernel/Regularizer/ConstЇ
dense_1/kernel/Regularizer/SumSum%dense_1/kernel/Regularizer/Square:y:0)dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/SumЙ
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2"
 dense_1/kernel/Regularizer/mul/xЉ
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/mulЙ
 dense_1/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_1/kernel/Regularizer/add/xє
dense_1/kernel/Regularizer/addAddV2)dense_1/kernel/Regularizer/add/x:output:0"dense_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/addЊ
IdentityIdentity%activation_3/PartitionedCall:output:0!^conv1d_1/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:€€€€€€€€€Ў::::2D
 conv1d_1/StatefulPartitionedCall conv1d_1/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:U Q
,
_output_shapes
:€€€€€€€€€Ў
!
_user_specified_name	input_2:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
©
Ї
E__inference_conv1d_1_layer_call_and_return_conditional_losses_2894128

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identityИp
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
conv1d/ExpandDims/dim†
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*9
_output_shapes'
%:#€€€€€€€€€€€€€€€€€€Ў2
conv1d/ExpandDimsє
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:Ў*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dimЄ
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:Ў2
conv1d/ExpandDims_1ј
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*8
_output_shapes&
$:"€€€€€€€€€€€€€€€€€€*
paddingVALID*
strides
2
conv1dТ
conv1d/SqueezeSqueezeconv1d:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€*
squeeze_dims
2
conv1d/SqueezeМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpХ
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2	
BiasAddq
IdentityIdentityBiasAdd:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*<
_input_shapes+
):€€€€€€€€€€€€€€€€€€Ў:::] Y
5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€Ў
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
Р'
Є
Y__inference_simple_convolutional_network_layer_call_and_return_conditional_losses_2894276
input_2
conv1d_1_2894157
conv1d_1_2894159
dense_1_2894249
dense_1_2894251
identityИҐ conv1d_1/StatefulPartitionedCallҐdense_1/StatefulPartitionedCallҐ!dropout_1/StatefulPartitionedCallъ
 conv1d_1/StatefulPartitionedCallStatefulPartitionedCallinput_2conv1d_1_2894157conv1d_1_2894159*
Tin
2*
Tout
2*+
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*N
fIRG
E__inference_conv1d_1_layer_call_and_return_conditional_losses_28941282"
 conv1d_1/StatefulPartitionedCallж
activation_2/PartitionedCallPartitionedCall)conv1d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*+
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*R
fMRK
I__inference_activation_2_layer_call_and_return_conditional_losses_28941672
activation_2/PartitionedCallл
max_pooling1d_1/PartitionedCallPartitionedCall%activation_2/PartitionedCall:output:0*
Tin
2*
Tout
2*+
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*U
fPRN
L__inference_max_pooling1d_1_layer_call_and_return_conditional_losses_28941472!
max_pooling1d_1/PartitionedCallф
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall(max_pooling1d_1/PartitionedCall:output:0*
Tin
2*
Tout
2*+
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*O
fJRH
F__inference_dropout_1_layer_call_and_return_conditional_losses_28941882#
!dropout_1/StatefulPartitionedCallџ
flatten_1/PartitionedCallPartitionedCall*dropout_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:€€€€€€€€€и* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*O
fJRH
F__inference_flatten_1_layer_call_and_return_conditional_losses_28942122
flatten_1/PartitionedCallМ
dense_1/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_1_2894249dense_1_2894251*
Tin
2*
Tout
2*'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_28942382!
dense_1/StatefulPartitionedCallб
activation_3/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*R
fMRK
I__inference_activation_3_layer_call_and_return_conditional_losses_28942592
activation_3/PartitionedCallµ
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_1_2894249*
_output_shapes
:	и*
dtype022
0dense_1/kernel/Regularizer/Square/ReadVariableOpі
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	и2#
!dense_1/kernel/Regularizer/SquareХ
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_1/kernel/Regularizer/ConstЇ
dense_1/kernel/Regularizer/SumSum%dense_1/kernel/Regularizer/Square:y:0)dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/SumЙ
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2"
 dense_1/kernel/Regularizer/mul/xЉ
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/mulЙ
 dense_1/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_1/kernel/Regularizer/add/xє
dense_1/kernel/Regularizer/addAddV2)dense_1/kernel/Regularizer/add/x:output:0"dense_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/addв
IdentityIdentity%activation_3/PartitionedCall:output:0!^conv1d_1/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:€€€€€€€€€Ў::::2D
 conv1d_1/StatefulPartitionedCall conv1d_1/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall:U Q
,
_output_shapes
:€€€€€€€€€Ў
!
_user_specified_name	input_2:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
≠
e
I__inference_activation_3_layer_call_and_return_conditional_losses_2894610

inputs
identityN
TanhTanhinputs*
T0*'
_output_shapes
:€€€€€€€€€2
Tanh\
IdentityIdentityTanh:y:0*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*&
_input_shapes
:€€€€€€€€€:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
ъ
J
.__inference_activation_3_layer_call_fn_2894615

inputs
identity•
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*R
fMRK
I__inference_activation_3_layer_call_and_return_conditional_losses_28942592
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*&
_input_shapes
:€€€€€€€€€:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
х'
ї
 __inference__traced_save_2894685
file_prefix.
*savev2_conv1d_1_kernel_read_readvariableop,
(savev2_conv1d_1_bias_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop'
#savev2_sgd_iter_read_readvariableop	(
$savev2_sgd_decay_read_readvariableop0
,savev2_sgd_learning_rate_read_readvariableop+
'savev2_sgd_momentum_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop
savev2_1_const

identity_1ИҐMergeV2CheckpointsҐSaveV2ҐSaveV2_1П
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
ConstН
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_9d7884f5f08d41f4af14a3c1e97be6e1/part2	
Const_1Л
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard¶
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameЕ
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:
*
dtype0*Ч
valueНBК
B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE2
SaveV2/tensor_namesЬ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:
*
dtype0*'
valueB
B B B B B B B B B B 2
SaveV2/shape_and_slices≈
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_conv1d_1_kernel_read_readvariableop(savev2_conv1d_1_bias_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop#savev2_sgd_iter_read_readvariableop$savev2_sgd_decay_read_readvariableop,savev2_sgd_learning_rate_read_readvariableop'savev2_sgd_momentum_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"/device:CPU:0*
_output_shapes
 *
dtypes
2
	2
SaveV2Г
ShardedFilename_1/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B :2
ShardedFilename_1/shardђ
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename_1Ґ
SaveV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2_1/tensor_namesО
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
SaveV2_1/shape_and_slicesѕ
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
_output_shapes
 *
dtypes
22

SaveV2_1г
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesђ
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

IdentityБ

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*I
_input_shapes8
6: :Ў::	и:: : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2SaveV22
SaveV2_1SaveV2_1:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:)%
#
_output_shapes
:Ў: 

_output_shapes
::%!

_output_shapes
:	и: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: "ѓL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*і
serving_default†
@
input_25
serving_default_input_2:0€€€€€€€€€Ў@
activation_30
StatefulPartitionedCall:0€€€€€€€€€tensorflow/serving/predict:к“
¶4
layer-0
layer_with_weights-0
layer-1
layer-2
layer-3
layer-4
layer-5
layer_with_weights-1
layer-6
layer-7
		optimizer

trainable_variables
regularization_losses
	variables
	keras_api

signatures
*`&call_and_return_all_conditional_losses
a__call__
b_default_save_signature"њ1
_tf_keras_model•1{"class_name": "Model", "name": "simple_convolutional_network", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "simple_convolutional_network", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 30, 600]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}, "name": "input_2", "inbound_nodes": []}, {"class_name": "Conv1D", "config": {"name": "conv1d_1", "trainable": true, "dtype": "float32", "filters": 24, "kernel_size": {"class_name": "__tuple__", "items": [2]}, "strides": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_1", "inbound_nodes": [[["input_2", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_2", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_2", "inbound_nodes": [[["conv1d_1", 0, 0, {}]]]}, {"class_name": "MaxPooling1D", "config": {"name": "max_pooling1d_1", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [1]}, "pool_size": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last"}, "name": "max_pooling1d_1", "inbound_nodes": [[["activation_2", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.1990393326736252, "noise_shape": null, "seed": null}, "name": "dropout_1", "inbound_nodes": [[["max_pooling1d_1", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_1", "inbound_nodes": [[["dropout_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_1", "inbound_nodes": [[["flatten_1", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_3", "trainable": true, "dtype": "float32", "activation": "tanh"}, "name": "activation_3", "inbound_nodes": [[["dense_1", 0, 0, {}]]]}], "input_layers": [["input_2", 0, 0]], "output_layers": [["activation_3", 0, 0]]}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 30, 600]}, "is_graph_network": true, "keras_version": "2.3.0-tf", "backend": "tensorflow", "model_config": {"class_name": "Model", "config": {"name": "simple_convolutional_network", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 30, 600]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}, "name": "input_2", "inbound_nodes": []}, {"class_name": "Conv1D", "config": {"name": "conv1d_1", "trainable": true, "dtype": "float32", "filters": 24, "kernel_size": {"class_name": "__tuple__", "items": [2]}, "strides": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_1", "inbound_nodes": [[["input_2", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_2", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_2", "inbound_nodes": [[["conv1d_1", 0, 0, {}]]]}, {"class_name": "MaxPooling1D", "config": {"name": "max_pooling1d_1", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [1]}, "pool_size": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last"}, "name": "max_pooling1d_1", "inbound_nodes": [[["activation_2", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.1990393326736252, "noise_shape": null, "seed": null}, "name": "dropout_1", "inbound_nodes": [[["max_pooling1d_1", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_1", "inbound_nodes": [[["dropout_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_1", "inbound_nodes": [[["flatten_1", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_3", "trainable": true, "dtype": "float32", "activation": "tanh"}, "name": "activation_3", "inbound_nodes": [[["dense_1", 0, 0, {}]]]}], "input_layers": [["input_2", 0, 0]], "output_layers": [["activation_3", 0, 0]]}}, "training_config": {"loss": "mean_squared_error", "metrics": null, "weighted_metrics": null, "loss_weights": null, "sample_weight_mode": null, "optimizer_config": {"class_name": "SGD", "config": {"name": "SGD", "learning_rate": 0.007431528531014919, "decay": 0.0, "momentum": 0.0, "nesterov": false}}}}
х"т
_tf_keras_input_layer“{"class_name": "InputLayer", "name": "input_2", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 30, 600]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 30, 600]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}}
Ї	

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
*c&call_and_return_all_conditional_losses
d__call__"Х
_tf_keras_layerы{"class_name": "Conv1D", "name": "conv1d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "conv1d_1", "trainable": true, "dtype": "float32", "filters": 24, "kernel_size": {"class_name": "__tuple__", "items": [2]}, "strides": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {"-1": 600}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 30, 600]}}
≤
trainable_variables
regularization_losses
	variables
	keras_api
*e&call_and_return_all_conditional_losses
f__call__"£
_tf_keras_layerЙ{"class_name": "Activation", "name": "activation_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "activation_2", "trainable": true, "dtype": "float32", "activation": "relu"}}
÷
trainable_variables
regularization_losses
	variables
	keras_api
*g&call_and_return_all_conditional_losses
h__call__"«
_tf_keras_layer≠{"class_name": "MaxPooling1D", "name": "max_pooling1d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "max_pooling1d_1", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [1]}, "pool_size": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}}
—
trainable_variables
regularization_losses
	variables
 	keras_api
*i&call_and_return_all_conditional_losses
j__call__"¬
_tf_keras_layer®{"class_name": "Dropout", "name": "dropout_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.1990393326736252, "noise_shape": null, "seed": null}}
√
!trainable_variables
"regularization_losses
#	variables
$	keras_api
*k&call_and_return_all_conditional_losses
l__call__"і
_tf_keras_layerЪ{"class_name": "Flatten", "name": "flatten_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "flatten_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
Х

%kernel
&bias
'trainable_variables
(regularization_losses
)	variables
*	keras_api
*m&call_and_return_all_conditional_losses
n__call__"р
_tf_keras_layer÷{"class_name": "Dense", "name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 360}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 360]}}
≤
+trainable_variables
,regularization_losses
-	variables
.	keras_api
*o&call_and_return_all_conditional_losses
p__call__"£
_tf_keras_layerЙ{"class_name": "Activation", "name": "activation_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "activation_3", "trainable": true, "dtype": "float32", "activation": "tanh"}}
I
/iter
	0decay
1learning_rate
2momentum"
	optimizer
<
0
1
%2
&3"
trackable_list_wrapper
'
q0"
trackable_list_wrapper
<
0
1
%2
&3"
trackable_list_wrapper
 
3layer_metrics
4metrics
5non_trainable_variables

trainable_variables

6layers
regularization_losses
7layer_regularization_losses
	variables
a__call__
b_default_save_signature
*`&call_and_return_all_conditional_losses
&`"call_and_return_conditional_losses"
_generic_user_object
,
rserving_default"
signature_map
&:$Ў2conv1d_1/kernel
:2conv1d_1/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
≠
8layer_metrics
9metrics
:non_trainable_variables
trainable_variables

;layers
regularization_losses
<layer_regularization_losses
	variables
d__call__
*c&call_and_return_all_conditional_losses
&c"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
≠
=layer_metrics
>metrics
?non_trainable_variables
trainable_variables

@layers
regularization_losses
Alayer_regularization_losses
	variables
f__call__
*e&call_and_return_all_conditional_losses
&e"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
≠
Blayer_metrics
Cmetrics
Dnon_trainable_variables
trainable_variables

Elayers
regularization_losses
Flayer_regularization_losses
	variables
h__call__
*g&call_and_return_all_conditional_losses
&g"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
≠
Glayer_metrics
Hmetrics
Inon_trainable_variables
trainable_variables

Jlayers
regularization_losses
Klayer_regularization_losses
	variables
j__call__
*i&call_and_return_all_conditional_losses
&i"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
≠
Llayer_metrics
Mmetrics
Nnon_trainable_variables
!trainable_variables

Olayers
"regularization_losses
Player_regularization_losses
#	variables
l__call__
*k&call_and_return_all_conditional_losses
&k"call_and_return_conditional_losses"
_generic_user_object
!:	и2dense_1/kernel
:2dense_1/bias
.
%0
&1"
trackable_list_wrapper
'
q0"
trackable_list_wrapper
.
%0
&1"
trackable_list_wrapper
≠
Qlayer_metrics
Rmetrics
Snon_trainable_variables
'trainable_variables

Tlayers
(regularization_losses
Ulayer_regularization_losses
)	variables
n__call__
*m&call_and_return_all_conditional_losses
&m"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
≠
Vlayer_metrics
Wmetrics
Xnon_trainable_variables
+trainable_variables

Ylayers
,regularization_losses
Zlayer_regularization_losses
-	variables
p__call__
*o&call_and_return_all_conditional_losses
&o"call_and_return_conditional_losses"
_generic_user_object
:	 (2SGD/iter
: (2	SGD/decay
: (2SGD/learning_rate
: (2SGD/momentum
 "
trackable_dict_wrapper
'
[0"
trackable_list_wrapper
 "
trackable_list_wrapper
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
q0"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ї
	\total
	]count
^	variables
_	keras_api"Д
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
:  (2total
:  (2count
.
\0
]1"
trackable_list_wrapper
-
^	variables"
_generic_user_object
≤2ѓ
Y__inference_simple_convolutional_network_layer_call_and_return_conditional_losses_2894458
Y__inference_simple_convolutional_network_layer_call_and_return_conditional_losses_2894303
Y__inference_simple_convolutional_network_layer_call_and_return_conditional_losses_2894276
Y__inference_simple_convolutional_network_layer_call_and_return_conditional_losses_2894496ј
Ј≤≥
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
∆2√
>__inference_simple_convolutional_network_layer_call_fn_2894384
>__inference_simple_convolutional_network_layer_call_fn_2894522
>__inference_simple_convolutional_network_layer_call_fn_2894344
>__inference_simple_convolutional_network_layer_call_fn_2894509ј
Ј≤≥
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
е2в
"__inference__wrapped_model_2894112ї
Л≤З
FullArgSpec
argsЪ 
varargsjargs
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *+Ґ(
&К#
input_2€€€€€€€€€Ў
Ш2Х
E__inference_conv1d_1_layer_call_and_return_conditional_losses_2894128Ћ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *+Ґ(
&К#€€€€€€€€€€€€€€€€€€Ў
э2ъ
*__inference_conv1d_1_layer_call_fn_2894138Ћ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *+Ґ(
&К#€€€€€€€€€€€€€€€€€€Ў
у2р
I__inference_activation_2_layer_call_and_return_conditional_losses_2894527Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
Ў2’
.__inference_activation_2_layer_call_fn_2894532Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
І2§
L__inference_max_pooling1d_1_layer_call_and_return_conditional_losses_2894147”
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *3Ґ0
.К+'€€€€€€€€€€€€€€€€€€€€€€€€€€€
М2Й
1__inference_max_pooling1d_1_layer_call_fn_2894153”
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *3Ґ0
.К+'€€€€€€€€€€€€€€€€€€€€€€€€€€€
 2«
F__inference_dropout_1_layer_call_and_return_conditional_losses_2894549
F__inference_dropout_1_layer_call_and_return_conditional_losses_2894544і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
Ф2С
+__inference_dropout_1_layer_call_fn_2894554
+__inference_dropout_1_layer_call_fn_2894559і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
р2н
F__inference_flatten_1_layer_call_and_return_conditional_losses_2894565Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
’2“
+__inference_flatten_1_layer_call_fn_2894570Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
о2л
D__inference_dense_1_layer_call_and_return_conditional_losses_2894596Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
”2–
)__inference_dense_1_layer_call_fn_2894605Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
у2р
I__inference_activation_3_layer_call_and_return_conditional_losses_2894610Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
Ў2’
.__inference_activation_3_layer_call_fn_2894615Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
і2±
__inference_loss_fn_0_2894628П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ 
4B2
%__inference_signature_wrapper_2894413input_2†
"__inference__wrapped_model_2894112z%&5Ґ2
+Ґ(
&К#
input_2€€€€€€€€€Ў
™ ";™8
6
activation_3&К#
activation_3€€€€€€€€€≠
I__inference_activation_2_layer_call_and_return_conditional_losses_2894527`3Ґ0
)Ґ&
$К!
inputs€€€€€€€€€
™ ")Ґ&
К
0€€€€€€€€€
Ъ Е
.__inference_activation_2_layer_call_fn_2894532S3Ґ0
)Ґ&
$К!
inputs€€€€€€€€€
™ "К€€€€€€€€€•
I__inference_activation_3_layer_call_and_return_conditional_losses_2894610X/Ґ,
%Ґ"
 К
inputs€€€€€€€€€
™ "%Ґ"
К
0€€€€€€€€€
Ъ }
.__inference_activation_3_layer_call_fn_2894615K/Ґ,
%Ґ"
 К
inputs€€€€€€€€€
™ "К€€€€€€€€€ј
E__inference_conv1d_1_layer_call_and_return_conditional_losses_2894128w=Ґ:
3Ґ0
.К+
inputs€€€€€€€€€€€€€€€€€€Ў
™ "2Ґ/
(К%
0€€€€€€€€€€€€€€€€€€
Ъ Ш
*__inference_conv1d_1_layer_call_fn_2894138j=Ґ:
3Ґ0
.К+
inputs€€€€€€€€€€€€€€€€€€Ў
™ "%К"€€€€€€€€€€€€€€€€€€•
D__inference_dense_1_layer_call_and_return_conditional_losses_2894596]%&0Ґ-
&Ґ#
!К
inputs€€€€€€€€€и
™ "%Ґ"
К
0€€€€€€€€€
Ъ }
)__inference_dense_1_layer_call_fn_2894605P%&0Ґ-
&Ґ#
!К
inputs€€€€€€€€€и
™ "К€€€€€€€€€Ѓ
F__inference_dropout_1_layer_call_and_return_conditional_losses_2894544d7Ґ4
-Ґ*
$К!
inputs€€€€€€€€€
p
™ ")Ґ&
К
0€€€€€€€€€
Ъ Ѓ
F__inference_dropout_1_layer_call_and_return_conditional_losses_2894549d7Ґ4
-Ґ*
$К!
inputs€€€€€€€€€
p 
™ ")Ґ&
К
0€€€€€€€€€
Ъ Ж
+__inference_dropout_1_layer_call_fn_2894554W7Ґ4
-Ґ*
$К!
inputs€€€€€€€€€
p
™ "К€€€€€€€€€Ж
+__inference_dropout_1_layer_call_fn_2894559W7Ґ4
-Ґ*
$К!
inputs€€€€€€€€€
p 
™ "К€€€€€€€€€І
F__inference_flatten_1_layer_call_and_return_conditional_losses_2894565]3Ґ0
)Ґ&
$К!
inputs€€€€€€€€€
™ "&Ґ#
К
0€€€€€€€€€и
Ъ 
+__inference_flatten_1_layer_call_fn_2894570P3Ґ0
)Ґ&
$К!
inputs€€€€€€€€€
™ "К€€€€€€€€€и<
__inference_loss_fn_0_2894628%Ґ

Ґ 
™ "К ’
L__inference_max_pooling1d_1_layer_call_and_return_conditional_losses_2894147ДEҐB
;Ґ8
6К3
inputs'€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ ";Ґ8
1К.
0'€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ ђ
1__inference_max_pooling1d_1_layer_call_fn_2894153wEҐB
;Ґ8
6К3
inputs'€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ ".К+'€€€€€€€€€€€€€€€€€€€€€€€€€€€ѓ
%__inference_signature_wrapper_2894413Е%&@Ґ=
Ґ 
6™3
1
input_2&К#
input_2€€€€€€€€€Ў";™8
6
activation_3&К#
activation_3€€€€€€€€€…
Y__inference_simple_convolutional_network_layer_call_and_return_conditional_losses_2894276l%&=Ґ:
3Ґ0
&К#
input_2€€€€€€€€€Ў
p

 
™ "%Ґ"
К
0€€€€€€€€€
Ъ …
Y__inference_simple_convolutional_network_layer_call_and_return_conditional_losses_2894303l%&=Ґ:
3Ґ0
&К#
input_2€€€€€€€€€Ў
p 

 
™ "%Ґ"
К
0€€€€€€€€€
Ъ »
Y__inference_simple_convolutional_network_layer_call_and_return_conditional_losses_2894458k%&<Ґ9
2Ґ/
%К"
inputs€€€€€€€€€Ў
p

 
™ "%Ґ"
К
0€€€€€€€€€
Ъ »
Y__inference_simple_convolutional_network_layer_call_and_return_conditional_losses_2894496k%&<Ґ9
2Ґ/
%К"
inputs€€€€€€€€€Ў
p 

 
™ "%Ґ"
К
0€€€€€€€€€
Ъ °
>__inference_simple_convolutional_network_layer_call_fn_2894344_%&=Ґ:
3Ґ0
&К#
input_2€€€€€€€€€Ў
p

 
™ "К€€€€€€€€€°
>__inference_simple_convolutional_network_layer_call_fn_2894384_%&=Ґ:
3Ґ0
&К#
input_2€€€€€€€€€Ў
p 

 
™ "К€€€€€€€€€†
>__inference_simple_convolutional_network_layer_call_fn_2894509^%&<Ґ9
2Ґ/
%К"
inputs€€€€€€€€€Ў
p

 
™ "К€€€€€€€€€†
>__inference_simple_convolutional_network_layer_call_fn_2894522^%&<Ґ9
2Ґ/
%К"
inputs€€€€€€€€€Ў
p 

 
™ "К€€€€€€€€€
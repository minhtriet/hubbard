import numpy as np
import pennylane as qp
from pennylane import QueuingManager
from pennylane.exceptions import MatrixUndefinedError, TransformError
from scipy import sparse
from functools import partial
from pennylane import transform
from pennylane.operation import Operator
from pennylane.pauli import PauliSentence, PauliWord
from pennylane.typing import TensorLike, PostprocessingFn


from pennylane.tape import QuantumScript, QuantumScriptBatch

from pennylane.ops.functions.matrix import catalyst_qjit, _matrix_transform


def matrix(
    op: Operator | PauliWord | PauliSentence, wire_order=None, return_sparse=False
) -> TensorLike:
    r"""The dense matrix representation of an operation or quantum circuit.

    .. note::
        This method always returns a dense matrix. For workflows with sparse objects, consider using :func:`~pennylane.operation.Operator.sparse_matrix`.

    Args:
        op (Operator or QNode or QuantumTape or Callable or PauliWord or PauliSentence): A quantum operator or quantum circuit.
        wire_order (Sequence[Any], optional): Order of the wires in the quantum circuit.
            The default wire order depends on the type of ``op``:

            - If ``op`` is a :class:`~.QNode`, then the wire order is determined by the
              associated device's wires, if provided.

            - Otherwise, the wire order is determined by the order in which wires
              appear in the circuit.

            - See the usage details for more information.

    Returns:
        TensorLike or qnode (QNode) or quantum function (Callable) or tuple[List[QuantumTape], function]:

        If an operator, :class:`~PauliWord` or :class:`~PauliSentence` is provided as input, the matrix is returned directly in the form of a tensor.
        Otherwise, the transformed circuit is returned as described in :func:`qml.transform <pennylane.transform>`.
        Executing this circuit will provide its matrix representation.

    **Example**

    Given an instantiated operator, ``qml.matrix`` returns the matrix representation:

    >>> op = qml.RX(0.54, wires=0)
    >>> qml.matrix(op)
    array([[0.9637709+0.j        , 0.       -0.26673144j],
           [0.       -0.26673144j, 0.9637709+0.j        ]])

    It can also be used in a functional form:

    >>> x = torch.tensor(0.6, requires_grad=True)
    >>> matrix_fn = qml.matrix(qml.RX)
    >>> matrix_fn(x, wires=0)
    tensor([[0.9553+0.0000j, 0.0000-0.2955j],
            [0.0000-0.2955j, 0.9553+0.0000j]], grad_fn=<StackBackward0>)

    In its functional form, it is fully differentiable with respect to gate arguments:

    >>> loss = torch.real(torch.trace(matrix_fn(x, wires=0)))
    >>> loss.backward()
    >>> x.grad
    tensor(-0.2955)

    This operator transform can also be applied to QNodes, tapes, and quantum functions
    that contain multiple operations; see Usage Details below for more details.

    .. details::
        :title: Usage Details

        ``qml.matrix`` can also be used with :class:`~PauliWord` and :class:`~PauliSentence` instances.
        Internally, we are using their ``to_mat()`` methods.

        >>> X0 = PauliWord({0:"X"})
        >>> np.allclose(qml.matrix(X0), X0.to_mat())
        True

        ``qml.matrix`` can also be used with QNodes, tapes, or quantum functions that
        contain multiple operations.

        Consider the following quantum function:

        .. code-block:: python

            def circuit(theta):
                qml.RX(theta, wires=1)
                qml.Z(0)

        We can use ``qml.matrix`` to generate a new function that returns the unitary matrix
        corresponding to the function ``circuit``:

        >>> matrix_fn = qml.matrix(circuit, wire_order=[1, 0])
        >>> theta = np.pi / 4
        >>> matrix_fn(theta)
        array([[ 0.92387953+0.j        ,  0.        +0.j        ,
             0.        -0.38268343j,  0.        +0.j        ],
           [ 0.        +0.j        , -0.92387953+0.j        ,
             0.        +0.j        ,  0.        +0.38268343j],
           [ 0.        -0.38268343j,  0.        +0.j        ,
             0.92387953+0.j        ,  0.        +0.j        ],
           [ 0.        +0.j        ,  0.        +0.38268343j,
             0.        +0.j        , -0.92387953+0.j        ]])

        You can also get the unitary matrix for operations on a subspace of a larger Hilbert space. For
        example, with the same function ``circuit`` and ``wire_order=["a", 0, "b", 1]`` you obtain the
        :math:`16\times 16` matrix for the operation :math:`I\otimes Z\otimes I\otimes  R_X(\theta)`.

        This unitary matrix can also be used in differentiable calculations. For example, consider the
        following cost function:

        .. code-block:: python

            def circuit(theta):
                qml.RY(theta, wires=0)

            def cost(theta):
                matrix = qml.matrix(circuit, wire_order=[0])(theta)
                return pnp.real(pnp.trace(matrix))

        Since this cost function returns a real scalar as a function of ``theta``, we can differentiate it:

        >>> theta = pnp.array(0.3, requires_grad=True)
        >>> # Expected value is 2 * cos(0.3 / 2)
        >>> cost(theta)
        np.float64(1.97...)
        >>> # The gradient is -sin(0.3 / 2)
        >>> qml.grad(cost, argnums=0)(theta)
        tensor(-0.14943813, requires_grad=True)

    """
    if catalyst_qjit(op):
        op = op.user_function

    if not isinstance(op, Operator):
        if isinstance(op, (PauliWord, PauliSentence)):
            if wire_order is None and len(op.wires) > 1:
                raise ValueError(
                    "wire_order is required by qml.matrix() for PauliWords "
                    "or PauliSentences with more than one wire."
                )
            return op.to_mat(wire_order=wire_order)

        if isinstance(op, QuantumScript):
            if wire_order is None:
                error_base_str = "wire_order is required by qml.matrix() for tapes"
                if len(op.wires) > 1:
                    raise ValueError(error_base_str + " with more than one wire.")
                if len(op.wires) == 0:
                    raise ValueError(error_base_str + " without wires.")

        elif isinstance(op, qp.QNode):
            if wire_order is None and op.device.wires is None:
                raise ValueError(
                    "wire_order is required by qml.matrix() for QNodes if the device does "
                    "not have wires specified."
                )

        elif callable(op):
            if getattr(op, "num_wires", 0) != 1 and wire_order is None:
                raise ValueError(
                    "wire_order is required by qml.matrix() for quantum functions."
                )

        else:
            raise TransformError(
                "Input is not an Operator, tape, QNode, or quantum function"
            )

        return _matrix_transform(op, wire_order=wire_order)

    # Starting from now, op is an Operator
    # Validate wire_order
    if wire_order and not set(op.wires).issubset(wire_order):
        raise TransformError(
            f"Wires in circuit {list(op.wires)} are inconsistent with "
            f"those in wire_order {list(wire_order)}"
        )
    QueuingManager.remove(op)
    if op.has_sparse_matrix and return_sparse:
        return sparse.csc_matrix(op.matrix(wire_order=wire_order))
    if op.has_matrix:
        return op.matrix(wire_order=wire_order)
    if op.has_decomposition:
        with QueuingManager.stop_recording():
            ops = op.decomposition()
        if return_sparse:
            return sparse.coo_matrix(
                matrix(QuantumScript(ops), wire_order=wire_order or op.wires)
            )
        return matrix(QuantumScript(ops), wire_order=wire_order or op.wires)
    raise MatrixUndefinedError(
        "Operator must define a matrix, sparse matrix, or decomposition for use with qml.matrix."
    )


@partial(transform, is_informative=True)
def _matrix_transform(
    tape: QuantumScript, wire_order=None, **kwargs
) -> tuple[QuantumScriptBatch, PostprocessingFn]:

    if wire_order and not set(tape.wires).issubset(wire_order):
        raise TransformError(
            f"Wires in circuit {list(tape.wires)} are inconsistent with "
            f"those in wire_order {list(wire_order)}"
        )

    wires = kwargs.get("device_wires", None) or tape.wires
    wire_order = wire_order or wires

    def processing_fn(res):
        """Defines how matrix works if applied to a tape containing multiple operations."""

        params = res[0].get_parameters(trainable_only=False)
        interface = qp.math.get_interface(*params)

        # initialize the unitary matrix
        if len(res[0].operations) == 0:
            result = qp.math.eye(2 ** len(wire_order), like=interface)
        else:
            result = matrix(res[0].operations[0], wire_order=wire_order)
        result = sparse.csr_matrix(result)

        for op in res[0].operations[1:]:
            U = matrix(op, wire_order=wire_order)
            # Coerce the matrices U and result and use matrix multiplication. Broadcasted axes
            # are handled correctly automatically by ``matmul`` (See e.g. NumPy documentation)
            result = U @ result

        return result

    return [tape], processing_fn


# test case 1
H = qp.X(0) @ qp.Y(1) + qp.Z(0) @ qp.Z(1)

dense = qp.matrix(qp.TrotterProduct(H, n=10, time=1.5, order=1))
sparse_to_dense = np.array(
    matrix(qp.TrotterProduct(H, n=10, time=1.5, order=1), return_sparse=True).todense()
)
np.allclose(dense, sparse_to_dense)

dense = qp.matrix(qp.TrotterProduct(H, n=10, time=1.5, order=2))
sparse_to_dense = np.array(
    matrix(qp.TrotterProduct(H, n=10, time=1.5, order=2), return_sparse=True).todense()
)
np.allclose(dense, sparse_to_dense)

coeffs = [0.25, 0.75]
ops = [qp.X(0), qp.Z(0)]
H = qp.dot(coeffs, ops)

dense = qp.matrix(qp.TrotterProduct(H, n=10, time=1.5, order=1))
sparse_to_dense = np.array(
    matrix(qp.TrotterProduct(H, n=10, time=1.5, order=1), return_sparse=True).todense()
)
np.allclose(dense, sparse_to_dense)


H = (
    -0.5 * (qp.Y(0) @ qp.Z(1) @ qp.Y(2))
    + -0.5 * (qp.X(0) @ qp.Z(1) @ qp.X(2))
    # + 8.0 * qp.I([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
    + -0.5 * (qp.Y(1) @ qp.Z(2) @ qp.Y(3))
    + -0.5 * (qp.X(1) @ qp.Z(2) @ qp.X(3))
    + -0.5 * (qp.Y(0) @ qp.Z(1) @ qp.Z(2) @ qp.Z(3) @ qp.Y(4))
    + -0.5 * (qp.X(0) @ qp.Z(1) @ qp.Z(2) @ qp.Z(3) @ qp.X(4))
    + -0.5 * (qp.Y(1) @ qp.Z(2) @ qp.Z(3) @ qp.Z(4) @ qp.Y(5))
    + -0.5 * (qp.X(1) @ qp.Z(2) @ qp.Z(3) @ qp.Z(4) @ qp.X(5))
    + -0.5
    * (
        qp.Y(0)
        @ qp.Z(1)
        @ qp.Z(2)
        @ qp.Z(3)
        @ qp.Z(4)
        @ qp.Z(5)
        @ qp.Z(6)
        @ qp.Z(7)
        @ qp.Y(8)
    )
    + -0.5
    * (
        qp.X(0)
        @ qp.Z(1)
        @ qp.Z(2)
        @ qp.Z(3)
        @ qp.Z(4)
        @ qp.Z(5)
        @ qp.Z(6)
        @ qp.Z(7)
        @ qp.X(8)
    )
    + -0.5
    * (
        qp.Y(1)
        @ qp.Z(2)
        @ qp.Z(3)
        @ qp.Z(4)
        @ qp.Z(5)
        @ qp.Z(6)
        @ qp.Z(7)
        @ qp.Z(8)
        @ qp.Y(9)
    )
    + -0.5
    * (
        qp.X(1)
        @ qp.Z(2)
        @ qp.Z(3)
        @ qp.Z(4)
        @ qp.Z(5)
        @ qp.Z(6)
        @ qp.Z(7)
        @ qp.Z(8)
        @ qp.X(9)
    )
)

num_wire = len(H.wires)
sparse_m = matrix(H, return_sparse=True)
assert sparse_m.shape == (2**num_wire, 2**num_wire)
np.allclose(qp.matrix(H), sparse_m.todense())

op = qp.TrotterProduct(H, n=10, time=1.5, order=1)
dense = qp.matrix(op)
sparse_m = matrix(op, return_sparse=True)
assert sparse_m.shape == (2**num_wire, 2**num_wire)
np.allclose(dense, sparse_m.todense())

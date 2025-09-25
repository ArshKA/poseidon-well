import torch

class _Reduce:
    @staticmethod
    def dims(x: torch.Tensor, reduce_dims: tuple[int, ...] | None) -> tuple[int, ...]:
        """
        If reduce_dims is None, reduce over all non-batch dims.
        """
        if reduce_dims is None:
            return tuple(range(1, x.ndim))
        return reduce_dims


class MSE:
    @staticmethod
    def eval(x: torch.Tensor, y: torch.Tensor, reduce_dims: tuple[int, ...] | None = None) -> torch.Tensor:
        dims = _Reduce.dims(x, reduce_dims)
        return torch.mean((x - y) ** 2, dim=dims)


class NMSE:
    @staticmethod
    def eval(
        x: torch.Tensor,
        y: torch.Tensor,
        eps: float = 1e-7,
        norm_mode: str = "norm",              # "norm" uses mean(y^2), "std" uses var(y)
        reduce_dims: tuple[int, ...] | None = None,
        unbiased: bool = True,                # mirror Well’s default behavior for std/var
    ) -> torch.Tensor:
        dims = _Reduce.dims(x, reduce_dims)
        if norm_mode == "norm":
            norm = torch.mean(y ** 2, dim=dims)
        elif norm_mode == "std":
            # Well uses std(y)^2 = var(y) with default unbiased behavior
            norm = torch.var(y, dim=dims, unbiased=unbiased)
        else:
            raise ValueError(f"Invalid norm_mode: {norm_mode}")
        return MSE.eval(x, y, reduce_dims=dims) / (norm + eps)


class NRMSE:
    @staticmethod
    def eval(
        x: torch.Tensor,
        y: torch.Tensor,
        eps: float = 1e-7,
        norm_mode: str = "norm",
        reduce_dims: tuple[int, ...] | None = None,
        unbiased: bool = True,
    ) -> torch.Tensor:
        return torch.sqrt(NMSE.eval(x, y, eps=eps, norm_mode=norm_mode, reduce_dims=reduce_dims, unbiased=unbiased))


class VRMSE:
    @staticmethod
    def eval(
        x: torch.Tensor,
        y: torch.Tensor,
        eps: float = 1e-7,
        reduce_dims: tuple[int, ...] | None = None,
        unbiased: bool = True,
    ) -> torch.Tensor:
        """
        Root Variance-Scaled MSE:
            sqrt( MSE(x,y) / ( var(y) + eps ) )
        where reductions are over `reduce_dims` (default: all non-batch dims).
        """
        return NRMSE.eval(x, y, eps=eps, norm_mode="std", reduce_dims=reduce_dims, unbiased=unbiased)

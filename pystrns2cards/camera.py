from typing import Tuple
import math
import torch
import torch.nn.functional as F

def cos_sin(theta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute cosine and sine of angles."""
    theta = theta.view(-1, 1)
    return torch.cos(theta), torch.sin(theta)

def rot_x(theta: torch.Tensor) -> torch.Tensor:
    """Rotation matrix around X-axis for given angles."""
    c, s = cos_sin(theta)
    m = torch.zeros(theta.numel(), 3, 3, dtype=theta.dtype, device=theta.device)
    m[..., 0, 0] = 1.0
    m[..., 1, 1], m[..., 1, 2] = c, -s
    m[..., 2, 1], m[..., 2, 2] = s, c
    return m

def rot_y(theta: torch.Tensor) -> torch.Tensor:
    """Rotation matrix around Y-axis for given angles."""
    c, s = cos_sin(theta)
    m = torch.zeros(theta.numel(), 3, 3, dtype=theta.dtype, device=theta.device)
    m[..., 1, 1] = 1.0
    m[..., 0, 0], m[..., 0, 2] = c, s
    m[..., 2, 0], m[..., 2, 2] = -s, c
    return m

def rot_z(theta: torch.Tensor) -> torch.Tensor:
    """Rotation matrix around Z-axis for given angles."""
    c, s = cos_sin(theta)
    m = torch.zeros(theta.numel(), 3, 3, dtype=theta.dtype, device=theta.device)
    m[..., 2, 2] = 1.0
    m[..., 0, 0], m[..., 0, 1] = c, -s
    m[..., 1, 0], m[..., 1, 1] = s, c
    return m

def euler_rotation(angles: torch.Tensor) -> torch.Tensor:
    """
    Create rotation matrices from Euler angles (degrees).
    Args:
        angles: Tensor of shape (B, 3), angles in degrees.
    Returns:
        Tensor of shape (B, 3, 3) representing rotation matrices.
    """
    theta = math.pi / 180.0 * angles
    return rot_z(theta[..., 2]) @ rot_y(theta[..., 1]) @ rot_x(theta[..., 0])

def perspective(fovy: torch.Tensor, aspect: torch.Tensor,
                near: torch.Tensor, far: torch.Tensor) -> torch.Tensor:
    """
    Create perspective projection matrices.
    Args:
        fovy: (B,) vertical field-of-view in degrees
        aspect: (B,) aspect ratio
        near: (B,) near plane distance
        far: (B,) far plane distance
    Returns:
        Tensor of shape (B, 4, 4)
    """
    B, device = fovy.shape[0], fovy.device
    tan_half_fovy = torch.tan(0.5 * math.pi * fovy / 180.0)
    r = aspect * near * tan_half_fovy
    t = near * tan_half_fovy

    m = torch.zeros(B, 4, 4, dtype=fovy.dtype, device=device)
    m[..., 0, 0] = near / r
    m[..., 1, 1] = near / t
    m[..., 2, 2] = -(far + near) / (far - near)
    m[..., 2, 3] = -2 * far * near / (far - near)
    m[..., 3, 2] = -1
    return m

def look_at(cam_pos: torch.Tensor, target: torch.Tensor,
            up: torch.Tensor) -> torch.Tensor:
    """
    Generate view matrices using look-at transformation.
    Args:
        cam_pos: (B, 3) camera positions
        target: (B, 3) target positions
        up: (B, 3) up directions
    Returns:
        Tensor of shape (B, 4, 4)
    """
    B, device = cam_pos.shape[0], cam_pos.device

    z = F.normalize(cam_pos - target, dim=-1)
    x = F.normalize(torch.cross(up, z, dim=-1), dim=-1)
    y = F.normalize(torch.cross(z, x, dim=-1), dim=-1)

    m = torch.zeros(B, 4, 4, dtype=cam_pos.dtype, device=device)
    m[..., 0, :3], m[..., 0, 3] = x, -torch.sum(x * cam_pos, dim=-1)
    m[..., 1, :3], m[..., 1, 3] = y, -torch.sum(y * cam_pos, dim=-1)
    m[..., 2, :3], m[..., 2, 3] = z, -torch.sum(z * cam_pos, dim=-1)
    m[..., 3, 3] = 1
    return m

def orthographic(
    left: torch.Tensor, right: torch.Tensor,
    bottom: torch.Tensor, top: torch.Tensor,
    near: torch.Tensor, far: torch.Tensor,
    flip_z: bool = True
) -> torch.Tensor:
    """
    Create orthographic projection matrices.
    Args:
        left, right, bottom, top, near, far: (B,)
        flip_z: whether to flip Z axis
    Returns:
        Tensor of shape (B, 4, 4)
    """
    B, device = left.shape[0], left.device
    sign = -1.0 if flip_z else 1.0

    m = torch.zeros(B, 4, 4, dtype=torch.float32, device=device)
    m[..., 0, 0] = 2 / (right - left)
    m[..., 1, 1] = 2 / (top - bottom)
    m[..., 2, 2] = sign * 2 / (far - near)
    m[..., 0, 3] = -(right + left) / (right - left)
    m[..., 1, 3] = -(top + bottom) / (top - bottom)
    m[..., 2, 3] = -sign * (far + near) / (far - near)
    m[..., 3, 3] = 1
    return m

class PerspectiveCamera:
    """
    Simple perspective camera abstraction.
    """
    def __init__(self,
                 fovy: torch.Tensor = torch.tensor([40.0], dtype=torch.float32),
                 aspect: torch.Tensor = torch.tensor([1.0], dtype=torch.float32),
                 near: torch.Tensor = torch.tensor([1e-2], dtype=torch.float32),
                 far: torch.Tensor = torch.tensor([100.0], dtype=torch.float32)) -> None:
        self.fovy = fovy.view(1).cpu()
        self.aspect = aspect.view(1).cpu()
        self.near = near.view(1).cpu()
        self.far = far.view(1).cpu()

    def look(self, origin: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
        """
        Create camera matrix from euler rotation.
        Args:
            origin: (B, 3) camera origins
            angles: (B, 3) Euler angles in degrees
        Returns:
            (B, 4, 4) camera matrices
        """
        origin = origin.view(-1, 3).cpu()
        angles = angles.view(-1, 3).cpu()
        R = euler_rotation(angles)
        up = R @ torch.tensor([[0, 1, 0]], dtype=torch.float32).T
        to = R @ torch.tensor([[0, 0, -1]], dtype=torch.float32).T
        return self.from_lookat(origin, origin + to.T, up.T)

    def from_lookat(self, origin: torch.Tensor, target: torch.Tensor,
                    up: torch.Tensor = torch.tensor([0, 1, 0], dtype=torch.float32)) -> torch.Tensor:
        """
        Create full camera matrix using look-at.
        Args:
            origin: (B, 3)
            target: (B, 3)
            up: (B, 3) or (3,)
        Returns:
            (B, 4, 4)
        """
        origin = origin.view(-1, 3).cpu()
        target = target.view(-1, 3).cpu()
        up = F.normalize(up.view(-1, 3)).cpu()
        return perspective(self.fovy, self.aspect, self.near, self.far) @ look_at(origin, target, up)

    def ndc2depth(self, pos: torch.Tensor) -> torch.Tensor:
        """
        Convert homogeneous position to Euclidean depth.
        Args:
            pos: (*, N, 4) homogeneous clip-space positions
        Returns:
            (*, N) linearized depth
        """
        tan_half_fovy = torch.tan(.5 * math.pi * self.fovy / 180.)
        mx = self.aspect * tan_half_fovy
        my = tan_half_fovy
        x, y, w = mx.to(pos.device) * pos[..., 0], my.to(pos.device) * pos[..., 1], pos[..., 3]
        return torch.sqrt(x ** 2 + y ** 2 + w ** 2)

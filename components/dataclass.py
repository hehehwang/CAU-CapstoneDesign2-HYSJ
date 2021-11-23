from typing import Union

import numpy as np
from multipledispatch import dispatch
from numpy.core.fromnumeric import prod


class Coord3d:
    def __init__(self, *args: Union[int, float, np.ndarray]):
        if len(args) == 1:
            self.__np = np.array(args[0], dtype=np.float64)
        elif len(args) == 3:
            self.__np = np.array(args, dtype=np.float64)
        else:
            raise ValueError("Invalid number of arguments")
        self.__x = self.__np[0]
        self.__y = self.__np[1]
        self.__z = self.__np[2]

    def unitVector(self) -> "Coord3d":
        ans = self.asNp / np.linalg.norm(self.asNp)
        return Coord3d(ans[0], ans[1], ans[2])

    def angleBtw(self, other: "Coord3d") -> float:
        unitVectorA = self.unitVector()
        unitVectorB = other.unitVector()
        product = np.dot(unitVectorA.asNp, unitVectorB.asNp)
        if product >= 1:
            return 0.0
        if product <= -1:
            return 180.0
        angleRad = np.arccos(product)
        return np.rad2deg(angleRad)

    def __eq__(self, __o: object) -> bool:
        if isinstance(__o, Coord3d):
            return np.isclose(self.asNp, __o.asNp).all()
        return False

    @property
    def x(self) -> np.float64:
        return self.__x

    @property
    def y(self) -> np.float64:
        return self.__y

    @property
    def z(self) -> np.float64:
        return self.__z

    @property
    def asNp(self) -> np.ndarray:
        return self.__np

    def __str__(self) -> str:
        return f"({self.__x}, {self.__y}, {self.__z})"

    def __repr__(self) -> str:
        return self.__str__()


class Landmark(Coord3d):
    def __init__(
        self,
        x: Union[int, float],
        y: Union[int, float],
        z: Union[int, float],
        vis: Union[int, float],
    ):
        super().__init__(x, y, z)
        self.__vis = vis

    @property
    def vis(self) -> Union[int, float]:
        return self.__vis

    def __str__(self) -> str:
        return f"({super().__str__()}): {self.__vis}"

    def __repr__(self) -> str:
        return self.__str__()

"""`sequence`: for making sequences to control PPMS.
"""
import csv
from typing import Any, Dict, Iterable, List, Tuple, TypeVar, Optional, Union

import numpy as np

SequenceCommand = TypeVar("SequenceCommand", bound="SequenceCommandBase")

class SequenceCommandBase:
    """Base class of 'SequenceCommand'.

    Attributes:
        commands (List[Tuple[Any]]): Commands of the sequence. The first element of the tuple indicates command type in integers (details below).
        command_number (Dict[str, int]):
            Dictionary of 'command name' to 'command number':
                `"Measure": 0,
                "WaitForField": 1,
                "WaitForTemp": 2,
                "SetField": 3,
                "SetTemp": 4,
                "SetPower": 5`
        field_approach_method_number (Dict[str, int]):
            Dictionary of 'field approach method' to the number:
                `"Linear": 0,
                "No overshoot": 1,
                "Oscillate": 2`
        magnet_mode_number (Dict[str, int]):
            Dictionary of 'magnet mode' to the number:
                `"Persistent": 0,
                "Driven": 1`
        temp_approach_method_number (Dict[str, int]):
            Dictionary of 'temperature approach method' to the number:
                `"Fast settle": 0,
                "No overshoot": 1`
    
    Methods:
        reset
            Note:
                reset 'commands' of the instance.
        to_csv
            Args:
                filename (str): File name of output csv file.
        calc_required_time
            Args:
                T0 (float): Temperature value before excuting the sequence (K). Defaults to 300.
                H0 (float): Magnetic field value before excuting the sequence (Oe). Defaults to 0.
                measuring_time (float): Required time of measuring sequence (min). Defaults to 6.
            
            Returns:
                (float): Required time of the entire sequence (min).
    
    Note:
        Additive operation (operator:`+`) is defined between this class and its child classes.
        The resulting instance of addition is always derived from `SequenceCommandBase`.
        Additive identity of this operation is `SequenceCommandBase()`.
    """
    def __init__(self) -> None:
        self.commands: List[Tuple[Any]] = []
        self._formatted_commands: List[str] = []

        # basically these are constant:
        self.command_number: Dict[str, int] = {
            "Measure": 0,
            "WaitForField": 1,
            "WaitForTemp": 2,
            "SetField": 3,
            "SetTemp": 4,
            "SetPower": 5,
        }
        self.field_approach_method_number: Dict[str, int] = {
            "Linear": 0,
            "No overshoot": 1,
            "Oscillate": 2,
        }
        self.magnet_mode_number: Dict[str, int] = {
            "Persistent": 0,
            "Driven": 1,
        }
        self.temp_approach_method_number: Dict[str, int] = {
            "Fast settle": 0,
            "No overshoot": 1,
        }
        

    def __str__(self) -> str:
        res: str = "\n".join([f"{i} {row}" for i, row in enumerate(self._formatted_commands)])
        res = res + f"\nRequired time: {self.calc_required_time():.5g} min (assuming initial (T,H) = (300K,0T) and necessary time for one measure = 6 min)"
        return res

    def __add__(self, other: SequenceCommand) -> SequenceCommand:
        res: SequenceCommand = SequenceCommandBase()
        res.commands = self.commands + other.commands
        res._formatted_commands = self._formatted_commands + other._formatted_commands
        return res

    def reset(self) -> None:
        self.commands = []
        self._formatted_commands = []

    
    def to_csv(self, filename: str) -> None:
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(self.commands)

    def calc_required_time(self, T0: float = 300., H0: float = 0., measuring_time: float = 6.) -> float:
        """Calculate required time of the sequence.

        Args:
            T0 (float): Temperature value before excuting the sequence (K). Defaults to 300.
            H0 (float): Magnetic field value before excuting the sequence (Oe). Defaults to 0.
            measuring_time (float): Required time of measuring sequence (min). Defaults to 6.
        
        Returns:
            (float): Required time of the entire sequence (min).
        """
        res: float = 0.
        num_measure: int = 0
        curT: float = T0
        curH: float = H0
        for com_num, *contents in self.commands:
            if com_num == self.command_number["Measure"]:
                num_measure += 1
            elif com_num == self.command_number["WaitForField"]:
                extra_wait, = contents
                res += extra_wait
            elif com_num == self.command_number["WaitForTemp"]:
                extra_wait, = contents
                res += extra_wait
            elif com_num == self.command_number["SetField"]:
                target, rate, app_num, mag_num = contents
                res += abs(target-curH) / abs(rate) / 60 # unit of 'rate' is [Oe/s]
                curH = target
            elif com_num == self.command_number["SetTemp"]:
                target, rate, app_num = contents
                res += abs(target-curT) / abs(rate) # unit of 'rate' is [K/min]
                curT = target
            elif com_num == self.command_number["SetPower"]:
                res += 0
            else:
                raise RuntimeError("Command number is invalid. Make sure it an integer 0-5")
        res += num_measure * measuring_time
        return res

class Measure(SequenceCommandBase):
    """Command Labview to enter measuring sequence.
    """
    def __init__(self) -> None:
        super().__init__()
        com_num: int = self.command_number["Measure"]
        self.commands = [(com_num,)]
        self._formatted_commands = ["Measure"]

class WaitForField(SequenceCommandBase):
    """Command PPMS to wait until the magnetic field stabilized.

    Args:
        extra_wait (float): Extra wait time after magnetic field stabilized (min).
    """
    def __init__(self,
            extra_wait: float,
        ) -> None:
        super().__init__()
        com_num: int = self.command_number["WaitForField"]
        self.commands = [(com_num, extra_wait)]
        self._formatted_commands = [f"WaitForField: extra wait {extra_wait:.6g} min"]

class WaitForTemp(SequenceCommandBase):
    """Command PPMS to wait until the temperature stabilized.

    Args:
        extra_wait (float): Extra wait time after temperature stabilized (min).
    """
    def __init__(self,
            extra_wait: float,
        ) -> None:
        super().__init__()
        com_num: int = self.command_number["WaitForTemp"]
        self.commands = [(com_num, extra_wait)]
        self._formatted_commands = [f"WaitForTemp: extra wait {extra_wait:.6g} min"]

class SetField(SequenceCommandBase):
    """Command PPMS to set magnetic field.

    Args:
        target (float): Target magnetic field value (Oe).
        rate (float): Speed of changing magnetic field (Oe/s). Defaults to 100.
        approach_method (str): Approach method of magnetic field to the target. Defaults to "Linear".
            ["Linear", "No overshoot", "Oscillate"] can be used.
        magnet_mode (str): Mode of magnet in PPMS. Defaults to "Persistent".
            ["Persistent", "Driven"] can be used.
    """
    def __init__(self,
            target: float,
            rate: float = 100,
            approach_method: str = "Linear",
            magnet_mode: str = "Persistent",
        ) -> None:
        super().__init__()
        com_num: int = self.command_number["SetField"]
        app_num: int = self.field_approach_method_number[approach_method]
        mag_num: int = self.magnet_mode_number[magnet_mode]
        self.commands = [(com_num, target, rate, app_num, mag_num)]
        self._formatted_commands = [f"SetField: {target:.6g} Oe, {approach_method}, {magnet_mode}"]

class SetTemp(SequenceCommandBase):
    """Command PPMS to set temperature.

    Args:
        target (float): Target temperature value (K).
        rate (float): Speed of changing temperature (K/min). Defaults to 5. Make sure that 0 <= rate <= 20.
        approach_method (str): Approach method of temperature to the target. Defaults to "Fast settle". 
            ["Fast settle", "No overshoot"] can be used.
    """
    def __init__(self,
            target: float,
            rate: float = 5,
            approach_method: str = "Fast settle",
        ) -> None:
        if not (0 <= rate <= 20):
            raise ValueError("'rate' must be in [0, 20]")
        super().__init__()
        com_num: int = self.command_number["SetTemp"]
        app_num: int = self.temp_approach_method_number[approach_method]
        self.commands = [(com_num, target, rate, app_num)]
        self._formatted_commands = [f"SetTemp: {target:.6g} K, {approach_method}"]

class SetPower(SequenceCommandBase):
    """Command K6221 to set heater power.

    Args:
        target: Target power value (mW).
    """
    def __init__(self,
            target: float,
        ) -> None:
        super().__init__()
        com_num: int = self.command_number["SetPower"]
        self.commands = [(com_num, target)]
        self._formatted_commands = [f"SetPower: {target:.6g} mW"]

class ScanField(SequenceCommandBase):
    """Command PPMS to scan magnetic field.

    Note:
        This sequence commands PPMS to wait for the magnetic field to stabilize each time it changes the value of the magnetic field.
    
    Args:
        start (float): Initial magnetic field value (Oe).
        end (float): Final magnetic field value (Oe).
        increment (float): Increment magnetic field value by a step (Oe).
        rate (float): Speed of changing magnetic field (Oe/s). Defaults to 100.
        approach_method (str): Approach method of magnetic field to the target. Defaults to "Linear".
            ["Linear", "No overshoot", "Oscillate"] can be used.
        magnet_mode (str): Mode of magnet in PPMS. Defaults to "Persistent".
            ["Persistent", "Driven"] can be used.
        substructure (Optional[SequenceCommand]): Commands to be executed while scanning for magnetic field. Defaults to None.
    """
    def __init__(self,
            start: float,
            end: float,
            increment: float,
            rate: float = 100,
            approach_method: str = "Linear",
            magnet_mode: str = "Persistent",
            substructure: Optional[SequenceCommand] = None
        ) -> None:
        super().__init__()
        com_num: int = self.command_number["SetField"]
        app_num: int = self.field_approach_method_number[approach_method]
        mag_num: int = self.magnet_mode_number[magnet_mode]
        steps: int = int(abs(end-start) / abs(increment)) + 1
        for v in np.linspace(start, end, steps):
            self.commands.extend(
                [
                    (com_num, v, rate, app_num, mag_num), # set field
                    (self.command_number["WaitForField"], 0) # wait for field stability
                ]
            )
            if substructure is not None:
                self.commands.extend(substructure.commands)
        self._formatted_commands = [
                f"ScanField: from {start:.6g} Oe to {end:.6g} Oe at {rate:.6g} Oe/s in {increment:.6g} Oe increments [{steps} steps], {approach_method}, {magnet_mode}"
            ]
        if substructure is not None:
            self._formatted_commands = self._formatted_commands + ["\t" + s for s in substructure._formatted_commands]

    @classmethod
    def from_field_list(cls,
            value_list: Iterable[float],
            rate: float = 100,
            approach_method: str = "Linear",
            magnet_mode: str = "Persistent",
            substructure: Optional[SequenceCommand] = None
        ) -> None:
        """
        Args:
            value_list (Iterable[float]): List of magnetic field (Oe).
            rate (float): Speed of changing magnetic field (Oe/s). Defaults to 100.
            approach_method (str): Approach method of magnetic field to the target. Defaults to "Linear".
                ["Linear", "No overshoot", "Oscillate"] can be used.
            magnet_mode (str): Mode of magnet in PPMS. Defaults to "Persistent".
                ["Persistent", "Driven"] can be used.
            substructure (Optional[SequenceCommand]): Commands to be executed while scanning for magnetic field. Defaults to None.
        
        Returns:
            (ScanField): Instance of 'ScanField'.
        """
        self = cls(0, 0, 1)
        self.reset()
        com_num: int = self.command_number["SetField"]
        app_num: int = self.field_approach_method_number[approach_method]
        mag_num: int = self.magnet_mode_number[magnet_mode]
        for v in value_list:
            self.commands.extend(
                [
                    (com_num, v, rate, app_num, mag_num), # set field
                    (self.command_number["WaitForField"], 0) # wait for field stability
                ]
            )
            if substructure is not None:
                self.commands.extend(substructure.commands)
        self._formatted_commands = [
                f"ScanField: from {value_list[0]:.6g} Oe to {value_list[-1]:.6g} Oe at {rate:.6g} Oe/s [{len(value_list)} steps], {approach_method}, {magnet_mode}"
            ]
        if substructure is not None:
            self._formatted_commands = self._formatted_commands + ["\t" + s for s in substructure._formatted_commands]
        return self
    
class ScanTemp(SequenceCommandBase):
    """Command PPMS to scan temperature.
    
    Note:
        This sequence commands PPMS to wait for the temperature to stabilize each time it changes the value of the temperature.
    
    Args:
        start (float): Initial temperature value (K).
        end (float): Final temperature value (K).
        increment (float): Increment temperature value by a step (K).
        rate (float): Speed of changing temperature (K/min). Defaults to 5. Make sure that 0 <= rate <= 20.
        approach_method (str): Approach method of temperature to the target. Defaults to "Fast settle".
            ["Fast settle", "No overshoot"] can be used.
        substructure (Optional[SequenceCommand]): Commands to be executed while scanning for temperature. Defaults to None.
    """
    def __init__(self,
            start: float,
            end: float,
            increment: float,
            rate: float = 5,
            approach_method: str = "Fast settle",
            substructure: Optional[SequenceCommand] = None
        ) -> None:
        if not (0 <= rate <= 20):
            raise ValueError("'rate' must be in [0, 20]")
        super().__init__()
        com_num: int = self.command_number["SetTemp"]
        app_num: int = self.temp_approach_method_number[approach_method]
        steps: int = int(abs(end-start) / abs(increment)) + 1
        for v in np.linspace(start, end, steps):
            self.commands.extend(
                [
                    (com_num, v, rate, app_num), # set temperature
                    (self.command_number["WaitForTemp"], 0) # wait for temperature stability
                ]
            )
            if substructure is not None:
                self.commands.extend(substructure.commands)
        self._formatted_commands = [
                f"ScanTemp: from {start:.6g} K to {end:.6g} K at {rate:.6g} K/min in {increment:.6g} K increments [{steps} steps], {approach_method}"
            ]
        if substructure is not None:
            self._formatted_commands = self._formatted_commands + ["\t" + s for s in substructure._formatted_commands]

    @classmethod
    def from_temp_list(cls,
            value_list: Iterable[float],
            rate: float = 5,
            approach_method: str = "Fast settle",
            substructure: Optional[SequenceCommand] = None
        ) -> None:
        """
        Args:
            value_list (Iterable[float]): List of temperature (K).
            rate (float): Speed of changing temperature (K/min). Defaults to 5. Make sure that 0 <= rate <= 20.
            approach_method (str): Approach method of temperature to the target. Defaults to "Fast settle".
                ["Fast settle", "No overshoot"] can be used.
            substructure (Optional[SequenceCommand]): Commands to be executed while scanning for temperature. Defaults to None.
        
        Returns:
            (ScanTemp): Instance of 'ScanTemp'.
        """
        if not (0 <= rate <= 20):
            raise ValueError("'rate' must be in [0, 20]")
        self = cls(0, 0, 1)
        self.reset()
        com_num: int = self.command_number["SetTemp"]
        app_num: int = self.temp_approach_method_number[approach_method]
        for v in value_list:
            self.commands.extend(
                [
                    (com_num, v, rate, app_num), # set temperature
                    (self.command_number["WaitForTemp"], 0) # wait for temperature stability
                ]
            )
            if substructure is not None:
                self.commands.extend(substructure.commands)
        self._formatted_commands = [
                f"ScanTemp: from {value_list[0]:.6g} K to {value_list[-1]:.6g} K at {rate:.6g} K/min [{len(value_list)} steps], {approach_method}"
            ]
        if substructure is not None:
            self._formatted_commands = self._formatted_commands + ["\t" + s for s in substructure._formatted_commands]
        return self

class ScanPower(SequenceCommandBase):
    """Command K6221 to scan power.
    
    Args:
        start (float): Initial power value (mW).
        end (float): Final power value (mW).
        increment (float): Increment power value by a step (mW).
        substructure (Optional[SequenceCommand]): Commands to be executed while scanning for power. Defaults to None.
    """
    def __init__(self,
            start: float,
            end: float,
            increment: float,
            substructure: Optional[SequenceCommand] = None
        ) -> None:
        super().__init__()
        com_num: int = self.command_number["SetPower"]
        steps: int = int(abs(end-start) / abs(increment)) + 1
        for v in np.linspace(start, end, steps):
            self.commands.append(
                (com_num, v)
            )
            if substructure is not None:
                self.commands.extend(substructure.commands)
        self._formatted_commands = [
                f"ScanPower: from {start:.6g} mW to {end:.6g} mW in {increment:.6g} K increments [{steps} steps]"
            ]
        if substructure is not None:
            self._formatted_commands = self._formatted_commands + ["\t" + s for s in substructure._formatted_commands]

    @classmethod
    def from_power_list(cls,
            value_list: Iterable[float],
            substructure: Optional[SequenceCommand] = None
        ) -> None:
        """
        Args:
            value_list (Iterable[float]): List of power (mW).
            substructure (Optional[SequenceCommand]): Commands to be executed while scanning for power. Defaults to None.
        
        Returns:
            (ScanPower): Instance of 'ScanPower'.
        """
        self = cls(0, 0, 1)
        self.reset()
        com_num: int = self.command_number["SetPower"]
        for v in value_list:
            self.commands.append(
                    (com_num, v)
                )
            if substructure is not None:
                self.commands.extend(substructure.commands)
        self._formatted_commands = [
                    f"ScanPower: from {value_list[0]:.6g} mW to {value_list[-1]:.6g} mW [{len(value_list)} steps]"
                ]
        if substructure is not None:
            self._formatted_commands = self._formatted_commands + ["\t" + s for s in substructure._formatted_commands]
        return self

def sequence_maker(filename: str, command_list: List[SequenceCommand]) -> SequenceCommandBase:
    """Make a sequence for controlling PPMS.

    Args:
        filename (str): File name.
        command_list (List[SequenceCommand]):
            List of instances of `SequenceCommand`.
            If you want to add some commands while `ScanField`, `ScanTemp` and `ScanPower`,
            you will use the argument `substructure` of those classes (the details are as follows).

    Example:
        >>> res = sequence_maker(filename="./a.csv", command_list=
            [
                SetPower(target=0.1),
                WaitForTemp(extra_wait=1),
                ScanTemp.from_temp_list([300,250,200,150,100,50,2],
                    substructure=WaitForTemp(extra_wait=5) + Measure()
                ),
                SetField(target=0, rate=100, approach_method="Linear", magnet_mode="Persistent"),
                ScanTemp(start=2, end=10, increment=2, rate=5,
                    substructure=ScanField(90000, -90000, 10000, rate=100, approach_method="Linear", magnet_mode="Persistent",
                        substructure=WaitForField(extra_wait=0.5) + Measure()
                    ) + ScanField(-90000, 90000, 10000, rate=100, approach_method="Linear", magnet_mode="Persistent", 
                        substructure=WaitForField(extra_wait=0.5) + Measure()
                    )
                ),
                ScanField.from_field_list([0, 1000, 2000, 3000, 5000, 10000, 90000],
                    substructure=WaitForField(extra_wait=0.5) + Measure()
                ),
                SetField(0),
                SetTemp(300)
            ]
        )

        >>> print(res)
        '''
        0 SetPower: 0.1 mW
        1 WaitForTemp: extra wait 1 min
        2 ScanTemp: from 300 K to 2 K at 5 K/min [7 steps], Fast settle
        3       WaitForTemp: extra wait 5 min
        4       Measure
        5 SetField: 0 Oe, Linear, Persistent
        6 ScanTemp: from 2 K to 10 K at 5 K/min in 2 K increments [5 steps], Fast settle
        7       ScanField: from 90000 Oe to -90000 Oe at 100 Oe/s in 10000 Oe increments [19 steps], Linear, Persistent
        8               WaitForField: extra wait 0.5 min
        9               Measure
        10      ScanField: from -90000 Oe to 90000 Oe at 100 Oe/s in 10000 Oe increments [19 steps], Linear, Persistent
        11              WaitForField: extra wait 0.5 min
        12              Measure
        13 ScanField: from 0 Oe to 90000 Oe at 100 Oe/s [7 steps], Linear, Persistent
        14      WaitForField: extra wait 0.5 min
        15      Measure
        16 SetField: 0 Oe, Linear, Persistent
        17 SetTemp: 300 K, Fast settle
        Required time: 1837.7 min (assuming initial (T,H) = (300K,0T) and necessary time for one measure = 6 min)
        '''

        >>> print(res.calc_required_time(T0=300, H0=0, measuring_time=10)) # necessary time for one measure = 10 min
        2653.7
    """
    res: SequenceCommandBase = sum(command_list, start=SequenceCommandBase())
    res.to_csv(filename)
    return res
    

def main():
    res = sequence_maker(filename="./a.csv", command_list=
        [
            SetPower(target=0.1),
            WaitForTemp(extra_wait=1),
            ScanTemp.from_temp_list([300,250,200,150,100,50,2],
                substructure=WaitForTemp(extra_wait=5) + Measure()
            ),
            SetField(target=0, rate=100, approach_method="Linear", magnet_mode="Persistent"),
            ScanTemp(start=2, end=10, increment=2, rate=5,
                substructure=ScanField(90000, -90000, 10000, rate=100, approach_method="Linear", magnet_mode="Persistent",
                    substructure=WaitForField(extra_wait=0.5) + Measure()
                ) + ScanField(-90000, 90000, 10000, rate=100, approach_method="Linear", magnet_mode="Persistent", 
                    substructure=WaitForField(extra_wait=0.5) + Measure()
                )
            ),
            ScanField.from_field_list([0, 1000, 2000, 3000, 5000, 10000, 90000],
                substructure=WaitForField(extra_wait=0.5) + Measure()
            ),
            SetField(0),
            SetTemp(300)
        ]
    )
    print(res)
    print(res.calc_required_time(T0=300, H0=0, measuring_time=10))

    # res = sequence_maker("./a.csv", [
    #             ScanField.from_field_list([0, 1000, 40000, 90000], rate=100,
    #                 substructure=WaitForField(extra_wait=0.5) + Measure()
    #                 )
    #         ])
    # print(res)

    # res = sequence_maker("./a.csv", [
    #             ScanTemp.from_temp_list([300, 150, 100], rate=5,
    #                 substructure=WaitForTemp(extra_wait=0.5) + Measure()
    #                 )
    #         ])
    # print(res)


if __name__ == "__main__":
    main()

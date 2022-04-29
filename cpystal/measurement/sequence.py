"""`sequence`: for making sequences to control PPMS.
"""
from typing import Any, Dict, Iterable, List, Tuple, TypeVar, Optional, Union

import csv
import numpy as np

SequenceCommand = TypeVar("SequenceCommand", bound="SequenceCommandBase")

class SequenceCommandBase:
    def __init__(self) -> None:
        self.commands: List[Tuple[Any]] = []
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
        self._formatted_commands: List[str] = []

    def __str__(self) -> str:
        return "\n".join([f"{i} {row}" for i, row in enumerate(self._formatted_commands)])

    def __add__(self, other: SequenceCommand) -> SequenceCommand:
        res: SequenceCommand = SequenceCommandBase()
        res.commands = self.commands + other.commands
        res._formatted_commands = self._formatted_commands + other._formatted_commands
        return res
    
    def to_csv(self, filename: str) -> None:
        with open(filename, 'w') as f:
            writer = csv.writer(f)
            writer.writerows(self.commands)

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
        target (float): Target field (Oe).
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
        target (float): Target temperature (K).
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
        target: Target power (mW).
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
    
    Args:
        *args (float, float, float):
            start (float): Initial magnetic field value (Oe).
            end (float): Final magnetic field value (Oe).
            increment (float): Increment magnetic field value by a step (Oe).
        *args (Iterable[float]):
            value_list (Iterable[float]): List of magnetic field (Oe).
        
        rate (float): Speed of changing magnetic field (Oe/s). Defaults to 100.
        approach_method (str): Approach method of magnetic field to the target. Defaults to "Linear".
            ["Linear", "No overshoot", "Oscillate"] can be used.
        magnet_mode (str): Mode of magnet in PPMS. Defaults to "Persistent".
            ["Persistent", "Driven"] can be used.
        substructure (Optional[SequenceCommand]): Commands to be executed while scanning for magnetic field. Defaults to None.
    """
    def __init__(self,
            *args: Any,
            rate: float = 100,
            approach_method: str = "Linear",
            magnet_mode: str = "Persistent",
            substructure: Optional[SequenceCommand] = None
        ) -> None:
        super().__init__()
        com_num: int = self.command_number["SetField"]
        app_num: int
        mag_num: int
        if len(args) == 1 and hasattr(args[0], "__iter__"):
            value_list: Iterable[float] = args[0]
            app_num = self.field_approach_method_number[approach_method]
            mag_num = self.magnet_mode_number[magnet_mode]
            for v in value_list:
                self.commands.append(
                    (com_num, v, rate, app_num, mag_num)
                )
                if substructure is not None:
                    self.commands.extend(substructure.commands)
            self._formatted_commands = [
                    f"ScanField: from {value_list[0]:.6g} Oe to {value_list[-1]:.6g} Oe at {rate:.6g} Oe/s [{len(value_list)} steps], {approach_method}, {magnet_mode}"
                ]
            if substructure is not None:
                self._formatted_commands = self._formatted_commands + ["\t" + s for s in substructure._formatted_commands]
        elif 3 <= len(args) <= 7:
            start: float
            end: float
            increment: float
            if len(args) == 3:
                start, end, increment = args
            elif len(args) == 4:
                start, end, increment, rate = args
            elif len(args) == 5:
                start, end, increment, rate, approach_method = args
            elif len(args) == 6:
                start, end, increment, rate, approach_method, magnet_mode = args
            else:
                start, end, increment, rate, approach_method, magnet_mode, substructure = args
            app_num = self.field_approach_method_number[approach_method]
            mag_num = self.magnet_mode_number[magnet_mode]
            steps: int = int(abs(end-start) / abs(increment)) + 1
            for v in np.linspace(start, end, steps):
                self.commands.append(
                    (com_num, v, rate, app_num, mag_num)
                )
                if substructure is not None:
                    self.commands.extend(substructure.commands)
            self._formatted_commands = [
                    f"ScanField: from {start:.6g} Oe to {end:.6g} Oe at {rate:.6g} Oe/s in {increment:.6g} Oe increments [{steps} steps], {approach_method}, {magnet_mode}"
                ]
            if substructure is not None:
                self._formatted_commands = self._formatted_commands + ["\t" + s for s in substructure._formatted_commands]

        else:
            raise TypeError("arguments are invalid")

class ScanTemp(SequenceCommandBase):
    """Command PPMS to scan temperature.
    
    Args:
        *args (float, float, float):
            start (float): Initial temperature value (K).
            end (float): Final temperature value (K).
            increment (float): Increment temperature value by a step (K).
        *args (Iterable[float]):
            value_list (Iterable[float]): List of temperature (K).

        rate (float): Speed of changing temperature (K/min). Defaults to 5. Make sure that 0 <= rate <= 20.
        approach_method (str): Approach method of temperature to the target. Defaults to "Fast settle".
            ["Fast settle", "No overshoot"] can be used.
        substructure (Optional[SequenceCommand]): Commands to be executed while scanning for temperature. Defaults to None.
    """
    def __init__(self,
            *args: Any,
            rate: float = 5,
            approach_method: str = "Fast settle",
            substructure: Optional[SequenceCommand] = None
        ) -> None:
        if not (0 <= rate <= 20):
            raise ValueError("'rate' must be in [0, 20]")
        super().__init__()
        com_num: int = self.command_number["SetTemp"]
        app_num: int
        if len(args) == 1 and hasattr(args[0], "__iter__"):
            value_list: Iterable[float] = args[0]
            app_num = self.temp_approach_method_number[approach_method]
            for v in value_list:
                self.commands.append(
                    (com_num, v, rate, app_num)
                )
                if substructure is not None:
                    self.commands.extend(substructure.commands)
            self._formatted_commands = [
                    f"ScanTemp: from {value_list[0]:.6g} K to {value_list[-1]:.6g} K at {rate:.6g} K/min [{len(value_list)} steps], {approach_method}"
                ]
            if substructure is not None:
                self._formatted_commands = self._formatted_commands + ["\t" + s for s in substructure._formatted_commands]
        elif 3 <= len(args) <= 6:
            start: float
            end: float
            increment: float
            if len(args) == 3:
                start, end, increment = args
            elif len(args) == 4:
                start, end, increment, rate = args
            elif len(args) == 5:
                start, end, increment, rate, approach_method = args
            else:
                start, end, increment, rate, approach_method, substructure = args
            app_num = self.temp_approach_method_number[approach_method]
            
            steps: int = int(abs(end-start) / abs(increment)) + 1
            for v in np.linspace(start, end, steps):
                self.commands.append(
                    (com_num, v, rate, app_num)
                )
                if substructure is not None:
                    self.commands.extend(substructure.commands)
            self._formatted_commands = [
                    f"ScanTemp: from {start:.6g} K to {end:.6g} K at {rate:.6g} K/min in {increment:.6g} K increments [{steps} steps], {approach_method}"
                ]
            if substructure is not None:
                self._formatted_commands = self._formatted_commands + ["\t" + s for s in substructure._formatted_commands]
        else:
            raise TypeError("arguments are invalid")
        


class ScanPower(SequenceCommandBase):
    """Command K6221 to scan power.
    
    Args:
        *args (float, float, float):
            start (float): Initial power value (mW).
            end (float): Final power value (mW).
            increment (float): Increment power value by a step (mW).
        *args (Iterable[float]):
            value_list (Iterable[float]): List of power (mW).

        substructure (Optional[SequenceCommand]): Commands to be executed while scanning for power. Defaults to None.
    """
    def __init__(self,
            *args:Any,
            substructure: Optional[SequenceCommand] = None
        ) -> None:
        super().__init__()
        com_num: int = self.command_number["SetPower"]
        if len(args) == 1 and hasattr(args[0], "__iter__"):
            value_list: Iterable[float] = args[0]
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
        elif 3 <= len(args) <= 4:
            start: float
            end: float
            increment: float
            if len(args) == 3:
                start, end, increment = args
            else:
                start, end, increment, substructure = args
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
        else:
            raise TypeError("arguments are invalid")
        

def SequenceMaker(command_list: List[SequenceCommand]) -> SequenceCommandBase:
    res: SequenceCommandBase = sum(command_list, start=SequenceCommandBase())
    return res
    

def main():
    res = SequenceMaker([
            WaitForTemp(extra_wait=1),
            ScanTemp([300,250,200,150,100,50,2],
                substructure=Measure()
            ),
            SetField(target=0, rate=100, approach_method="Linear", magnet_mode="Persistent"),
            ScanTemp(2, 10, 2, rate=5,
                substructure=ScanField(70000, -70000, 10000, rate=100, approach_method="Linear", magnet_mode="Persistent", substructure=Measure()
                ) + ScanField(-70000, 70000, 10000, rate=100, approach_method="Linear", magnet_mode="Persistent", substructure=Measure())
            ),
        ])
    print(res)
    # res.to_csv("/Users/ut/Desktop/a.csv")

if __name__ == "__main__":
    main()

"""
These are a bunch of example functions to generate circuits.
Add more as needed.
"""

import stim # Quantum circuit simulator


def get_builtin_circuit(builtin_name, distance, rounds, **noise_args):
  """
  Wrapper around the stim.Circuit.generated function to generate a circuit.
  - Arguments:
    builtin_name: Name of the builtin circuit to generate.
    distance: Distance of the code.
    rounds: Number of rounds of the circuit.
    noise_args: Noise parameters to use.
  - Return type:
    stim.Circuit
  """
  return stim.Circuit.generated(
    code_task = builtin_name, distance = distance, rounds = rounds,
    **noise_args
  )


def get_circuit_with_inserted_measurements(reference_circuit):
  """
  get_circuit_with_inserted_measurements: Take a circuit and insert data qubit measurements after each stabilizer measurement round.
  In order to do so, we first move the reset operations and one round of noiseless stabilizer measurements to the beginning of the circuit.
  After, a noiseless pass on data qubits is inserted. At this point, the circuit state is fully determined,
  and we can identify any changes in the data qubit readouts by simply comparing to a readout in a previous round.
  While one would not want to measure data qubits until the very end of a quantum algorithm,
  this would be one way to determine the changes in the state of data qubits after each round
  (immediately after each stabilizer measurement) for the purpose of training a NN.
  Arguments:
  - reference_circuit: A stim.Circuit object.
  Return type:
  - The modified stim.Circuit object.
    Note that the circuit will not contain detector and observable information.
  - The indices of measurement readouts needed to construct the observables.
  """
  scirc = str(reference_circuit).split("\n")
  forbidden = ["DETECTOR", "OBSERVABLE", "TICK", "SHIFT"]
  forbidden_ext = ["DEPOLARIZE", "ERROR", "QUBIT"]
  forbidden_ext.extend(forbidden)
  begin = False
  reset_lines = []
  round_lines = []
  obs_idxs = []
  dqubit_measure = None
  for lll in scirc:
    skip = "REPEAT" in lll or lll.startswith('}')
    line = lll.strip()
    for f in forbidden_ext:
      if f in line:
        skip = True
        if f == "OBSERVABLE":
          for rr in line.split(" ")[1:]:
            rr = rr.replace("[", "").replace("]", "").replace("rec", "")
            obs_idxs.append(int(rr))
        break
    if skip:
      continue
    is_reset = False
    for sb in ["R", "RX", "RY", "RZ"]:
      if line.startswith(sb):
        reset_lines.append(line)
        begin = True
        is_reset = True
        break
    is_end = False
    if begin and not is_reset:
      round_lines.append(line)
      if line.startswith("MR"):
        begin = False
        is_end = True
    if is_end:
      continue
    if dqubit_measure is None:
      for sb in ["MX", "MY", "MZ", "M"]:
        if line.startswith(sb) and line != round_lines[-1]:
          dqubit_measure = line
          break
  round_lines.append(dqubit_measure)

  mod_scirc = []
  primordial_written = False
  for lll in scirc:
    line = lll.strip()
    wsp = lll.replace(line, "")
    skip = False
    for f in forbidden:
      if f in line:
        skip = True
        break
    if skip:
      continue
    is_reset = False
    for sb in ["R", "RX", "RY", "RZ"]:
      if line.startswith(sb) and "REPEAT" not in line:
        is_reset = True
        break
    if is_reset:
      if not primordial_written:
        mod_scirc.extend(reset_lines)
        mod_scirc.extend(round_lines)
        primordial_written = True
      continue
    mod_scirc.append(lll)
    if line == round_lines[-2]:
      mod_scirc.append(wsp+round_lines[-1])

  return stim.Circuit("\n".join(mod_scirc)), obs_idxs


def get_custom_circuit(circuit):
  """
  Wrapper to simply pick an external circuit.
  - Arguments:
    circuit: A stim.Circuit object.
  - Return type:
    stim.Circuit
  """
  return circuit


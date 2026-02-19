"""
MoeLens Exporter
================
Füge diesen Exporter in dein Training ein, um nach dem Training
die Daten für das MoeLens-Dashboard zu exportieren.

Verwendung:
    exporter = MoeLensExporter(placement_manager, layer_id=0)
    exporter.record_comm_matrix(step, comm_matrix)  # in jedem Forward-Pass
    exporter.save("moelens_layer0.json")            # am Ende des Trainings
"""

import json
import torch
from typing import Optional


class MoeLensExporter:
    """
    Sammelt Daten aus dem ExpertPlacementManager und exportiert
    sie als JSON für das MoeLens-Dashboard.
    """

    def __init__(self, placement_manager, layer_id: int = 0, total_steps: int = 0):
        self.manager    = placement_manager
        self.layer_id   = layer_id
        self.total_steps = total_steps
        self.comm_matrices = []   # {step, matrix}

    def record_comm_matrix(self, step: int, comm_matrix: torch.Tensor):
        """
        Aufruf in MOELayer.forward() nach dem comm_matrix-Block:

            exporter.record_comm_matrix(self.global_step, comm_matrix)

        Speichert jede 50. Matrix, um die Datei klein zu halten.
        """
        if step % 50 != 0:
            return
        self.comm_matrices.append({
            "step": step,
            "matrix": comm_matrix.cpu().tolist(),
        })

    def save(self, path: str):
        """Exportiert alle gesammelten Daten als JSON."""
        m   = self.manager
        nE  = m.num_experts
        nG  = m.ep_size

        # Rebalancing-History aus dem Manager
        history = []
        for event in m.rebalance_history:
            history.append({
                "step":          event["step"],
                "init_cost":     round(event.get("init_cost", 0), 2),
                "final_cost":    round(event.get("final_cost", 0), 2),
                "improve_pct":   round(event.get("improve_pct", 0), 1),
                "swaps":         event.get("swaps", 0),
                "iters":         event.get("iters", 0),
                # Snapshot der Platzierung – du kannst diese vor/nach dem
                # Rebalancing im Manager separat speichern falls nötig
                "placement_before": event.get("placement_before", [e//( nE//nG) for e in range(nE)]),
                "placement_after":  event.get("placement_after",  m.placement),
                "migrations":       event.get("migrations", []),
            })

        # Affinitätsmatrix (letzte gespeicherte)
        aff = m.accumulator.get().tolist()

        data = {
            "meta": {
                "num_experts": nE,
                "num_gpus":    nG,
                "layer_id":    self.layer_id,
                "total_steps": self.total_steps,
            },
            "rebalance_history": history,
            "comm_matrices":     self.comm_matrices,
            "affinity_matrix":   aff,
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        print(f"[MoeLens] Exportiert: {path}  ({len(history)} Events, {len(self.comm_matrices)} Matrizen)")
        return path


# ── Beispiel-Integration in MOELayer ──────────────────────────
#
# In MOELayer.__init__:
#   from .moelens_exporter import MoeLensExporter
#   self.moelens = MoeLensExporter(self.placement_manager, layer_id=layer_i)
#
# In MOELayer.forward() direkt nach dem comm_matrix-Block:
#   self.moelens.record_comm_matrix(self.placement_manager.global_step, comm_matrix)
#
# Nach dem Training (z.B. in deinem Trainings-Script):
#   model.moe_layers[0].moelens.save("moelens_layer0.json")
#   # Dann moelens_layer0.json in das Dashboard laden
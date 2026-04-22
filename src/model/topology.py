from email import header

from sortedcontainers import SortedDict
from collections import defaultdict
import pynvml
import csv
import torch
from torch.distributed.device_mesh import *
import numpy as np

from filelock import FileLock


class Device:
    def __init__(self, id, name):
        self.id = id
        self.name = name
        self.neighbors: dict[str, list[Device]] = defaultdict(list)

    def __repr__(self):
        return f"Device(id={self.id}, name='{self.name}')"

    def __eq__(self, other):
        if not isinstance(other, Device):
            return NotImplemented
        return self.__hash__() == other.__hash__()

    def __lt__(self, other):
        if not isinstance(other, Device): 
            return NotImplemented
        return self.id < other.id

    def __hash__(self):
        return hash((self.id, self.name))
    
    def toTorchDevice(self):
        return torch.device("cuda", self.id)


class Link:
    # TODO: raise N/A and NULL exceptions 
    def __init__(self, device_a_id:int, device_b_id:int, link_type:str, bandwidth:int, latency:int):
        if device_a_id is None or device_b_id is None or link_type is None or bandwidth is None or latency is None:
            raise TypeError(f"Link cannot be created with a None value! device_a: {device_a_id}, device_b: {device_b_id}, link_type: {link_type}, bandwidth: {bandwidth}, latency: {latency}")
        self.devices = tuple(sorted([device_a_id, device_b_id]))
        self.link_type = link_type
        self.bandwidth = bandwidth
        self.latency = latency

    def __repr__(self):
        return (f"Link({self.devices[0]} <-> {self.devices[1]}, "
                f"type={self.link_type}, bw={self.bandwidth}, lat={self.latency})")

    def __lt__(self, other):
        if not isinstance(other, Link):
            return NotImplemented
        return (-self.bandwidth if self.bandwidth is not None else float("inf"), self.latency if self.latency is not None else float("inf")) < (-other.bandwidth if other.bandwidth is not None else float("inf"), other.latency if other.latency is not None else float("inf"))

    def __eq__(self, other):
        if not isinstance(other, Link):
            return NotImplemented
        return self.__hash__() == other.__hash__()

    def __hash__(self):
        return hash((self.devices, self.link_type, self.bandwidth, self.latency))


class Topology:
    def __init__(self):
        self.devices: dict[int, Device] = {}
        self.links_by_bw: dict[int, list[Link]] = SortedDict()
        self.link_map: dict[tuple[int, int], Link] = {}
        self.bw_matrix: np.ndarray = None
        self.lat_matrix: np.ndarray = None

    def add_device(self, name, id=None):
        if id is None:
            devices_len = len(self.devices)
            self.devices[devices_len] = Device(devices_len, name)
            return self.devices[devices_len]
        else:
            if id not in self.devices:
                self.devices[id] = Device(id, name)
            return self.devices[id]
    
    def add_link(self, device_a, device_b, link_type, bandwidth, latency):
        link = Link(device_a.id, device_b.id, link_type, bandwidth, latency)
        key = link.devices

        # Update neighbors
        if device_b.id not in device_a.neighbors[link_type]:
            device_a.neighbors[link_type].append(device_b.id)
        if device_a.id not in device_b.neighbors[link_type]:
            device_b.neighbors[link_type].append(device_a.id)

        self.link_map[key] = link

        # Correct links_by_bw handling
        bw = int(bandwidth)
        if bw not in self.links_by_bw:
            self.links_by_bw[bw] = []
        if link not in self.links_by_bw[bw]:
            self.links_by_bw[bw].append(link)

        return link
    
    def build_matrices(self):
        """
        Build bandwidth and latency matrices for fast lookup.
        Call this after all devices and links are added.
        """
        n_devices = len(self.devices)
        
        # Initialize matrices with worst-case values
        self.bw_matrix = np.zeros((n_devices, n_devices), dtype=float)
        self.lat_matrix = np.full((n_devices, n_devices), float('inf'), dtype=float)
        
        # Diagonal entries (same device = infinite bandwidth, zero latency)
        for i in range(n_devices):
            self.bw_matrix[i, i] = float('inf')
            self.lat_matrix[i, i] = 0.0
        
        # Fill from link_map
        for (dev_a, dev_b), link in self.link_map.items():
            bw = float(link.bandwidth) if link.bandwidth > 0 else 1e-9
            lat = float(link.latency) if link.latency > 0 else float('inf')
            
            # Symmetric matrix
            self.bw_matrix[dev_a, dev_b] = bw
            self.bw_matrix[dev_b, dev_a] = bw
            self.lat_matrix[dev_a, dev_b] = lat
            self.lat_matrix[dev_b, dev_a] = lat
        
        print(f"Built {n_devices}x{n_devices} bandwidth and latency matrices")
        return self.bw_matrix, self.lat_matrix
    
    def get_bandwidth(self, dev_a: int, dev_b: int) -> float:
        """Get bandwidth between two devices."""
        if self.bw_matrix is None:
            self.build_matrices()
        return self.bw_matrix[dev_a, dev_b]
    
    def get_latency(self, dev_a: int, dev_b: int) -> float:
        """Get latency between two devices."""
        if self.lat_matrix is None:
            self.build_matrices()
        return self.lat_matrix[dev_a, dev_b]
    
    def getHierarchyLevels(self):
        return list(self.links_by_bw.keys()) 

    def getCliquesAtLevel(self, h):
        return self.links_by_bw[self.links_by_bw.keys()[h]]

    def __repr__(self):
        return f"Topology(devices={list(self.devices.values())}, links_by_bw={self.links_by_bw})"


def scan_topology(output_file="topology.csv"):
    print("\n=== Starting topology scan ===")
    pynvml.nvmlInit()
    device_count = pynvml.nvmlDeviceGetCount()
    print(f"Detected {device_count} GPU devices")
    
    with open(output_file, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["GPU_A", "GPU_B", "LinkType", "Bandwidth (MB/s)", "Latency (ns)"])
        
        for i in range(device_count):
            handle_i = pynvml.nvmlDeviceGetHandleByIndex(i)
            name_i = str(i) + "_" + pynvml.nvmlDeviceGetName(handle_i)
            print(f"\nScanning links for GPU {i}: {name_i}")

            for j in range(device_count):
                if i == j:
                    continue
                handle_j = pynvml.nvmlDeviceGetHandleByIndex(j)
                name_j = str(j) + "_" + pynvml.nvmlDeviceGetName(handle_j)
                print(f"  Checking connection to GPU {j}: {name_j}")
                
                link_type = pynvml.nvmlDeviceGetTopologyCommonAncestor(handle_i, handle_j)
                print(f"    Topology type: {link_type}")
                
                if link_type == pynvml.NVML_TOPOLOGY_INTERNAL:
                    link_str = "Internal"
                elif link_type == pynvml.NVML_TOPOLOGY_SINGLE:
                    link_str = "Single PCIe"
                elif link_type == pynvml.NVML_TOPOLOGY_MULTIPLE:
                    link_str = "Multiple PCIe"
                elif link_type == pynvml.NVML_TOPOLOGY_HOSTBRIDGE:
                    link_str = "HostBridge"
                elif link_type == pynvml.NVML_TOPOLOGY_NODE:
                    link_str = "Node"
                else:
                    link_str = "Unknown"

                print(f"    Link type string: {link_str}")
                
                for link in range(pynvml.NVML_NVLINK_MAX_LINKS):
                    try:
                        if pynvml.nvmlDeviceGetNvLinkState(handle_i, link):
                            print(f"    Found active NVLink {link}")
                            remote_pci = pynvml.nvmlDeviceGetNvLinkRemotePciInfo(handle_i, link)
                            handle_remote = pynvml.nvmlDeviceGetHandleByPciBusId(remote_pci.busId.decode())

                            if handle_remote == handle_j:
                                bw = pynvml.nvmlDeviceGetNvLinkThroughput(handle_i, link, pynvml.NVML_NVLINK_THROUGHPUT_DATA)
                                latency = pynvml.nvmlDeviceGetNvLinkUtilizationCounter(handle_i, link, 0)[0]
                                print(f"      NVLink details - BW: {bw} MB/s, Latency: {latency} ns")
                                writer.writerow([name_i, name_j, "NVLink", bw, latency])
                    except pynvml.NVMLError as e:
                        print(f"    NVLink error on link {link}: {str(e)}")
                        continue

                writer.writerow([name_i, name_j, link_str, "N/A", "N/A"])
    
    print("\nTopology scan completed")
    pynvml.nvmlShutdown()

def load_topology_from_csv(csv_file):
    print(f"\n=== Loading topology from {csv_file} ===")
    topology = Topology()
    
    with open(csv_file, newline="") as f:
        reader = csv.reader(f, delimiter=',')
        next(reader, None)
        
        link_count = 0
        for row in reader:
            if not row:
                continue
            dev_a_str, dev_b_str, link_type, bandwidth, latency = row
            
            id_a, name_a = dev_a_str.split("_", 1)
            id_b, name_b = dev_b_str.split("_", 1)
            
            device_a = topology.add_device(name_a, int(id_a))
            device_b = topology.add_device(name_b, int(id_b))
            
            try:
                bandwidth = int(bandwidth) if bandwidth.strip() != "N/A" else 0
                latency = int(latency) if latency.strip() != "N/A" else 0
            except ValueError:
                bandwidth = 0
                latency = 0
            
            link = topology.add_link(device_a, device_b, link_type, bandwidth, latency)
            if link:
                link_count += 1
            
        print(f"\nLoaded {len(topology.devices)} devices and {link_count} links")
    
    # Build matrices immediately after loading
    topology.build_matrices()
    
    return topology

def get_topology(file = "src/topology.csv"):
    lock = FileLock("src/topology.csv.lock")

    with lock:
        topo = load_topology_from_csv(file)
    # device_count = torch.cuda.device_count()

    # if len(topo.devices) > device_count:
    #     print("Topology has more devices than GPUs, rescanning...")
    #     scan_topology(file)
    #     topo = load_topology_from_csv(file)

    if len(topo.devices) == 0:
        print("No devices found, falling back to single-GPU topology.")
        topo = Topology()
        device = topo.add_device("GPU", 0)
        topo.add_link(device, device, "Self", 16000, 100)

    return topo

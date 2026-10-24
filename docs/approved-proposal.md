# Approved competition proposal — text reference

This is a text extraction of the owner's approved proposal. The [original architecture figure](images/proposal-architecture.jpg) is published
separately; PDF metadata is excluded. The source describes planned scope; measured current
behavior and limitations are recorded separately in the README and evidence register.

OpenCV AI Competition 2026, powered by AWS
CARE-Vision
Adaptive Agentic Vision for Safer Independent Living
Competition paths: COOL + Agentic Vision
Team: CARE-Vision Team
Member: Victor Ojeda - Argentina

CARE-Vision  ·  OpenCV AI Competition 2026
PROJECT SUMMARY
CARE-Vision is an agentic active-perception system for safer independent living. Its core computer vision 
workload runs on AWS Graviton using OpenCV 5 + COOL. A cloud-based Agentic Controller evaluates visual 
uncertainty and dynamically controls the effective frame transmission rate, resolution, and region of 
interest requested from the Edge Gateway. A local ring buffer enables retrieval of pre-event evidence when 
additional context is needed. This creates a closed perception-decision-action loop while reducing 
unnecessary bandwidth and computation and supporting network-aware resilience for deployment inside 
real homes.
1. Problem and Real-World Impact
Millions of older adults want to maintain their independence and continue living safely in their own homes. However, 
events such as falls, prolonged inactivity, or unusual movement can require rapid assistance, while continuous human 
supervision is impractical and continuous high-resolution video monitoring raises privacy, bandwidth, and cost 
concerns.
CARE-Vision is a computer vision system designed to detect and intelligently verify potentially dangerous situations 
involving older adults living independently. Instead of continuously analyzing video at maximum quality, CARE-Vision 
uses active perception: the system evaluates what it currently sees, estimates uncertainty, and decides whether it 
needs additional visual evidence before taking action.
The goal is to provide caregivers with timely and reliable incident information while minimizing unnecessary video 
transmission and cloud processing.
2. Target Users and Beneficiaries
• Older adults living independently.
• Family members and caregivers.
• Assisted-living and home-care providers.
Note: the system is intended as a safety-support and incident-detection system, not as a medical diagnostic device.
3. OpenCV 5 Video Analysis
OpenCV 5 will provide the core computer vision pipeline. Planned analysis includes person detection and tracking, 
pose/posture analysis, motion analysis, activity recognition, fall/incident detection, temporal analysis across multiple 
frames, and scene/context analysis.
Rather than classifying individual frames independently, CARE-Vision maintains temporal context to distinguish 
normal activities — such as sitting, bending down, or lying down — from potentially dangerous events.
The core vision workload will execute on AWS Graviton using OpenCV 5 and COOL.
Page 2

CARE-Vision  ·  OpenCV AI Competition 2026
4. Agentic Vision and Active Perception
PERCEIVE  →  UNDERSTAND  →  DECIDE  →  ACT  →  PERCEIVE AGAIN
CARE-Vision follows a closed perception-decision-action loop. OpenCV 5 + COOL produces structured observations. 
An Agentic Controller running on AWS Graviton maintains temporal state, evaluates uncertainty, and selects the next 
action:
• Keep observing or change the effective frame transmission rate.
• Increase or decrease transmitted resolution.
• Request a specific Region of Interest (ROI).
• Request higher-quality visual evidence.
• Change camera viewpoint when supported.
• Continue monitoring, trigger an incident notification, or return to low-consumption observation.
Capture rate vs. transmission rate
The camera can capture locally at a stable rate (for example, 30 FPS), while the Edge Gateway dynamically 
samples how many of those frames are transmitted to AWS. A command such as SET_TRANSMIT_FPS(20) 
means that approximately 20 frames per second are selected from the locally captured stream and sent to 
Graviton — it does not require physically reconfiguring the camera frame rate.
For example, unusual downward motion can cause the agent to request additional frames. If evidence remains 
ambiguous, it can request a higher-resolution person ROI. That new evidence is processed again by OpenCV 5 + 
COOL, completing the agentic loop.
5. Event-Driven Streaming
CARE-Vision uses an event-driven streaming strategy rather than continuously transmitting maximum-quality video. 
The camera may capture locally at a stable rate while the Edge Gateway maintains a short high-density ring buffer 
and controls the effective transmission rate to AWS.
The Edge Gateway handles video capture, frame sampling, a short rolling buffer, encoding/compression, network 
monitoring, store-and-forward, and execution of camera/stream commands. It does not perform the primary 
computer vision workload.
State Observation policy Example transmission
NORMAL Low observation 2 FPS, reduced resolution
SUSPICIOUS Increased observation 10 FPS, 720p
UNCERTAIN / VERIFYING Targeted observation 20 FPS, high-quality person ROI + 
recent buffer
INCIDENT Maximum relevant evidence Up to 30 FPS, high quality + alert
Page 3

CARE-Vision  ·  OpenCV AI Competition 2026
State Observation policy Example transmission
RECOVERY Return to low observation 2 FPS, reduced resolution
Note: the numeric rates above are initial example policies and will be calibrated experimentally. The important control 
variable is the effective frame transmission rate, not necessarily the physical capture FPS of the camera.
Because the Edge Gateway retains a short ring buffer, the agent can also request recent pre-event evidence. For 
example, after detecting an ambiguous event at time T, the cloud can request the previous five seconds at a denser 
sampling rate, allowing the vision pipeline to analyze what happened immediately before the suspected incident.
This approach reduces unnecessary bandwidth, cloud computation, and exposure of private visual information while 
retaining access to high-quality evidence when needed.
6. AWS Architecture and Planned Services
The architecture is divided between a lightweight home edge and an AWS cloud core.
Edge Gateway
• Capture video locally at a stable camera rate.
• Maintain a short high-density rolling buffer.
• Sample frames according to the transmission policy received from the agent.
• Resize/crop the requested representation or ROI and encode/compress it.
• Monitor network conditions and transmit adaptive video evidence.
• Execute sensing commands received from the cloud.
• Store and forward event evidence during connectivity interruptions.
AWS Cloud
Service Role in CARE-Vision
AWS Graviton OpenCV 5 + COOL core vision pipeline and Agentic 
Controller.
Amazon S3 Controlled evaluation data, selected encrypted event 
evidence, and benchmark artifacts.
Amazon DynamoDB System state, event metadata, and temporal state 
information.
Amazon SQS Event/message decoupling where required.
Amazon CloudWatch CPU, latency, errors, throughput, and benchmark 
monitoring.
The architecture deliberately places the core computer vision workload on Graviton using COOL, while the edge is 
responsible primarily for acquisition, buffering, frame sampling, transport, and actuation.
Page 4

CARE-Vision  ·  OpenCV AI Competition 2026
7. High-Level Architecture
Figure 1. CARE-Vision high-level architecture: Edge acquisition and resilience, Graviton-hosted OpenCV 5 + COOL core vision, and the 
adaptive agentic perception loop.
Page 5

CARE-Vision  ·  OpenCV AI Competition 2026
8. Network Resilience
Home Internet connectivity cannot be assumed to be perfect, so CARE-Vision explicitly models network conditions.
GOOD NETWORK
• Complete active-perception cycle operates normally.
DEGRADED NETWORK
• Adapt FPS, resolution, compression, and ROI selection to available bandwidth and latency.
OFFLINE
• Detect connectivity loss, continue local buffering, store relevant evidence, attempt reconnection, and 
synchronize buffered evidence when connectivity returns.
A lightweight local safety fallback may be evaluated for critical offline situations; it does not replace the primary 
Graviton/COOL workload. The system will explicitly indicate degraded operation rather than silently behaving as if 
cloud analysis were available.
9. COOL Path
CARE-Vision will follow the COOL path. COOL will execute the project's claimed core computer vision workload on 
AWS Graviton. The evaluation will first compare a baseline implementation against Graviton + COOL under equivalent 
sensing conditions. A second experiment will introduce Active Perception to measure additional resource savings.
Configuration Purpose
Baseline Reference performance and resource consumption
Graviton + COOL Measure COOL/Arm gains with equivalent sensing
Graviton + COOL + Active Perception Measure combined vision and adaptive-sensing 
efficiency
Metrics will include FPS, processing latency, CPU utilization, memory consumption, throughput, compute cost, and 
cost per processed workload.
10. Agentic Vision Path
CARE-Vision will also follow the Agentic Vision path. The agent does not simply describe what the camera sees: it 
directly controls subsequent perception.
Page 6

CARE-Vision  ·  OpenCV AI Competition 2026
Step Detail
Observation Possible fall, confidence 0.68.
Decision Additional evidence is required.
Action Request SET_TRANSMIT_FPS(20) and a higher-
resolution person ROI.
Edge execution
Sample ~20 frames per second from the locally 
captured stream, crop/encode the requested ROI, and 
transmit it.
New perception OpenCV 5 + COOL analyzes the new evidence.
Decision Confirm incident or return to normal monitoring.
This perception-decision-action feedback loop is a core architectural component of CARE-Vision.
11. Evaluation Plan
Area Proposed Evaluation
Computer Vision Precision, recall, F1, false-positive and false-negative 
rates.
Temporal Incident detection, confirmation, and recovery 
latency.
COOL / Cloud FPS, CPU, memory, processing latency, throughput, 
compute cost.
Active Perception
Fixed vs adaptive transmission: frames 
sent/processed, data transmitted, compute, detection 
performance.
Network Resilience Normal, reduced bandwidth, increased latency, 
packet loss, and complete outage.
Page 7

CARE-Vision  ·  OpenCV AI Competition 2026
12. Judge Demonstration
The final demonstration will show an end-to-end safety scenario and make the agent's decisions visible to the judges:
• Normal activity: camera captures locally while the edge transmits only the low-observation sample.
• Suspicious movement: OpenCV 5 + COOL identifies an ambiguous event.
• Active perception: the agent requests a higher effective transmission rate, resolution, ROI, and when useful 
recent pre-event frames from the ring buffer.
• Incident confirmation: the person remains on the floor without recovery; the agent confirms the event and 
generates a caregiver notification.
• Recovery: if normal movement resumes, the system returns to low-observation transmission.
The demo will display current vision state, agent decision, capture FPS versus transmitted FPS, resolution/ROI, 
network usage, processing latency, and COOL/Graviton metrics.
13. Expected Results
• Working OpenCV 5 computer vision pipeline.
• Verified execution of the core workload using COOL on AWS Graviton.
• Functioning Agentic Vision perception-decision-action loop.
• Adaptive effective frame transmission rate, resolution, and ROI control.
• Ring-buffer based pre-event evidence retrieval.
• Event-driven video streaming.
• Network degradation and offline resilience tests.
• COOL versus baseline benchmarks.
• Fixed versus Active Perception benchmarks.
• Complete end-to-end judge demonstration.
14. Team Profile
Victor Ojeda — Argentina
Software engineer and university lecturer with experience building production software and working 
with GPU/HPC infrastructure. Current focus is Computer Vision and Production AI, working with 
OpenCV, PyTorch, ONNX, Python, and cloud technologies.
Participating individually and responsible for the complete CARE-Vision system, including the OpenCV 5 
+ COOL pipeline on AWS Graviton, active perception, agentic workflow, deployment, benchmarking, 
and evaluation.
Previously participated in the OpenCV PyTorch Segmentation Project — Round 3, gaining hands-on 
experience building and evaluating computer vision models in a competitive setting.
Page 8

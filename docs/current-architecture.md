# Current architecture and agentic workflow

Selected demonstration profile, 21 October 2026. Capture and transmission rates
are independent. The primary inference and deterministic controller remain on AWS
Graviton. Other AWS services listed in the proposal are not claimed as deployed.

```mermaid
flowchart LR
 C[Camera: nominal 25 FPS] --> E[Edge: acquisition, buffer, sampling, JPEG]
 E -->|SSH tunnel: full frame 1280x720, requested 2/5 FPS| V
 subgraph G[AWS Graviton m8g.4xlarge]
 V[OpenCV 5 + COOL: YOLOX-S, RTMPose, temporal evidence] --> O[Structured observation]
 O --> A[Deterministic agent]
 A -->|SET_TRANSMIT_FPS| E
 E -->|ACK and newly captured evidence| V
 V --> P[Image-free portal projection]
 A --> P
 end
 J[Authenticated judge browser] --> F[CloudFront HTTPS]
 F -->|Restricted VPC origin| P
```

```mermaid
sequenceDiagram
 participant E as Edge
 participant V as Graviton OpenCV 5 + COOL
 participant A as Agent
 E->>V: Baseline visual evidence
 V->>A: Posture change with uncertainty
 A->>E: SET_TRANSMIT_FPS(5)
 E->>V: Applied ACK and new 5 FPS evidence
 V->>A: Updated posture/temporal observations
 A->>E: SET_TRANSMIT_FPS(2)
 V->>A: Later sustained seated pattern
 Note over A: Experimental resolution to NORMAL
```

ROI/buffer actions exist in other profiles; they are not exercised in this trial.
Alerts are disabled. Refer to technical-report.md for measurements and limitations.

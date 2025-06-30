"""
Creating Down Path

channels = [128, 256, 256, 512]

🟩 First Iteration
    c = 128, ch = 128, res = 64 → 32
    ResidualBlock(128, 128, 256) added
    res = 32 → not in attn_res → no attention blok
    Downsample(128) added
    c becomes 128

🟩 Second Iteration
    c = 128, ch = 256, res = 32 → 16
    ResidualBlock(128, 256, 256) added
    res = 16 → in attn_res → AttentionBlock(256) added
    Downsample(256) added
    c becomes 256

🟩 Third Iteration
    c = 256, ch = 256, res = 16 → 8
    ResidualBlock(256, 256, 256) added
    res = 8 → in attn_res → AttentionBlock(256) added
    Downsample(256) added
    c remains 256

🟩 Fourth Iteration
    c = 256, ch = 512, res = 8 → 4
    ResidualBlock(256, 512, 256) added
    res = 4 → not in attn_res
    Downsample(512) added
    c becomes 512

✅ Final content:
self.down_blocks (in order):
    [
    ResidualBlock(128, 128, 256)
    ResidualBlock(128, 256, 256)
    AttentionBlock(256)
    ResidualBlock(256, 256, 256)
    AttentionBlock(256)
    ResidualBlock(256, 512, 256)
    ]


self.downs:
    [
    Downsample(128)
    Downsample(256)
    Downsample(256)
    Downsample(512)
    ]

"""

"""
For upsampling path

First Iteration
    c = 512, ch = 512, res = 4 → 8
    Upsample(512)
    ResidualBlock(512 + 512, 512, 256) → ResidualBlock(1024, 512, 256)
    res = 8 ∈ attn_res → AttentionBlock(512)
    c becomes 512

2️⃣ Second Iteration
    c = 512, ch = 256, res = 8 → 16
    Upsample(512)
    ResidualBlock(512 + 256, 256, 256) → ResidualBlock(768, 256, 256)
    res = 16 ∈ attn_res → AttentionBlock(256)
    c becomes 256

3️⃣ Third Iteration
    c = 256, ch = 256, res = 16 → 32
    Upsample(256)
    ResidualBlock(256 + 256, 256, 256) → ResidualBlock(512, 256, 256)
    res = 32 ∉ attn_res → no AttentionBlock
    c remains 256

4️⃣ Fourth Iteration
    c = 256, ch = 128, res = 32 → 64
    Upsample(256)
    ResidualBlock(256 + 128, 128, 256) → ResidualBlock(384, 128, 256)
    res = 64 ∉ attn_res → no AttentionBlock
    c becomes 128

self.ups:
    [
    Upsample(512),
    Upsample(512),
    Upsample(256),
    Upsample(256)
    ]

self.up_blocks":
    [
    ResidualBlock(1024, 512, 256),
    AttentionBlock(256),
    ResidualBlock(768, 256, 256),
    AttentionBlock(256),
    ResidualBlock(512, 256, 256),
    ResidualBlock(384, 128, 256)
    ]
"""


"""
In usual Unet architecture diagram you will find crop and copy in skip connections but this cropping
is not needed here because we are making the spatial size of residual block of encoder 
exactly equal to spatial size in corresponding decoder residual block
"""

"""

Tensor Size Flow Visualization
A Input 
    Input: [8, 3, 64, 64]
    ↓ input_conv → [8, 128, 64, 64]

B DownPath
    1. ResidualBlock(128 → 128): [8, 128, 64, 64] (skip)
    2. Downsample: [8, 128, 32, 32]

    3. ResidualBlock(128 → 256): [8, 256, 32, 32] (skip)
    4. AttentionBlock(256): [8, 256, 32, 32]
    5. Downsample: [8, 256, 16, 16]

    6. ResidualBlock(256 → 256): [8, 256, 16, 16] (skip)
    7. AttentionBlock(256): [8, 256, 16, 16]
    8. Downsample: [8, 256, 8, 8]

    9. ResidualBlock(256 → 512): [8, 512, 8, 8] (skip)
    10. Downsample: [8, 512, 4, 4]

C Mid Block/Bottlenect
    11. ResidualBlock(512 → 512): [8, 512, 4, 4]
    12. AttentionBlock(512): [8, 512, 4, 4]
    13. ResidualBlock(512 → 512): [8, 512, 4, 4]

D Up Path   
    14. Upsample: [8, 512, 8, 8]
    15. Concat with skip [ResidualBlock from step 9 → 512]: [8, 1024, 8, 8]
    16. ResidualBlock(1024 → 512): [8, 512, 8, 8]

    17. Upsample: [8, 512, 16, 16]
    18. Concat with skip [ResidualBlock from step 6 → 256]: [8, 768, 16, 16]
    19. ResidualBlock(768 → 256): [8, 256, 16, 16]
    20. AttentionBlock(256): [8, 256, 16, 16]

    21. Upsample: [8, 256, 32, 32]
    22. Concat with skip [ResidualBlock from step 3 → 256]: [8, 512, 32, 32]
    23. ResidualBlock(512 → 256): [8, 256, 32, 32]
    24. AttentionBlock(256): [8, 256, 32, 32]

    25. Upsample: [8, 256, 64, 64]
    26. Concat with skip [ResidualBlock from step 1 → 128]: [8, 384, 64, 64]
    27. ResidualBlock(384 → 128): [8, 128, 64, 64]

E Output
    Final projection: [8, 3, 64, 64]
"""


"""
During experimentation of mnist it is 28x28 so it will be slight modification in code and now 

# ------------------ Down Path ------------------
self.down_blocks = [
    ResidualBlock(64, 64, 128),
    ResidualBlock(64, 128, 128),
    AttentionBlock(128),
]
self.downs = [
    Downsample(64),
    Downsample(128),
]

# ------------------ Middle ------------------
self.mid1 = ResidualBlock(128, 128, 128)
self.mid_attn = AttentionBlock(128)
self.mid2 = ResidualBlock(128, 128, 128)

# ------------------ Up Path ------------------
self.ups = [
    Upsample(128),
    Upsample(128),
]
self.up_blocks = [
    ResidualBlock(256, 128, 128),
    AttentionBlock(128),
    ResidualBlock(192, 64, 128),
]

"""
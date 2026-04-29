#import "../template.typ": problem, deliverable, note, section-label, subhead, prompt-box, answer-box, plot, answer

#problem("leaderboard", "Leaderboard: Fastest Training Step", "10 points")[
#prompt-box[
#section-label[Prompt]

Report your best result for the leaderboard benchmark described in the handout: batch size 2 on two B200 GPUs, measuring a complete training step from an empty PyTorch/Triton cache.

#deliverable[Your best wall-clock time for a full forward-and-backward training step with AdamW.]
]

#answer()
]

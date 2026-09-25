# Limitations and broader impact (draft text for the paper)

Written in the paper's style (STYLE_GUIDE.md): present tense for what the paper shows, past
tense for what we did, "we" as the subject, no dashes as punctuation.

## Limitations

**Synthetic scenes.** All results are on synthetic scenes: CLEVRER, ComPhy and CoPhy. CLEVRER
has three shapes, one object size, a flat table and almost no occlusion (two masks overlap
by more than 30% in 0.019% of pair-frames). We did not test the engine on real video, where
tracking, occlusion and contact are harder.

**Perception with labels.** Our tracks come from the detections released with NS-DR, whose
Mask R-CNN was trained with CLEVRER's object masks and attributes. The fair comparison group is
therefore the methods that use such labels (VRDP†‡ and CRCG), and the unsupervised results of
ALOE are a different setting.

**Evidence we cannot check.** In CLEVRER only the recorded world has ground truth. An answer
decided inside the simulated part of a counterfactual world (41% of correct answers) can be
inspected but not verified. CoPhy supplies the true counterfactual run, and there we report
event precision and recall; CLEVRER does not.

**Planar physics.** The simulators move objects on a plane and do not model toppling, falling
or 3D rotation. We therefore did not run CoPhy's CollisionCF and BlocktowerCF.

**Settings tuned to the answer key.** The hand-off look-ahead and the extension past the video
end were chosen on VAL-A. The extension helps because CLEVRER's answers include collisions after
the last frame (34.6% of the questions whose removed object never collides need one). The
setting is fitted to how the key was made.

**Language.** The question parser covers CLEVRER's templates (100% exact match on
counterfactual programs). Free-form questions need a language model; we tested one only in the
interactive demo.

**Official test set.** CLEVRER's and ComPhy's test servers closed before this work
(16 January 2026 and 31 January 2026). Our test sets are held-out validation and training
videos that were never used for training or selection, so comparisons with published test
numbers are approximate.

**Compute.** Every experiment ran on one laptop CPU (Intel i5-8350U, 4 cores, 7.9 GB RAM) with
no GPU. This kept the learned world model small (43,460 parameters) and ruled out large
pretrained video models as simulators. It limited us to two extra training seeds, each stopped
after two of four epochs. It also shows that the full method, including training, runs on
ordinary hardware: training took 8.1 hours and answering takes 2.2 seconds per question.

## Broader impact

The engine answers "what if" questions by simulating an intervention and reports the event
behind each answer, so a user can check why an answer was given. This makes errors visible:
in our audit, 98.2% of the checkable correct answers rest on a real recorded event, and the
wrong answers split into detector errors and simulation errors that can be traced one by one.

The same property can be misread. An answer that comes with a frame number and a named
collision looks trustworthy even when the deciding event was simulated and cannot be verified.
Our interface marks every event as recorded or simulated for this reason, and any use of the
method outside synthetic scenes should keep that distinction.

The data are synthetic and contain no people. CLEVRER-Humans contains human causal judgments
collected and released by its authors; we used the released labels only. We used a free
language model service (Groq) to link CLEVRER-Humans event sentences to objects; its outputs
are cached in the repository.

We do not see a direct route to misuse. The method needs object tracks and a simulator of the
scene, which limits it to settings where the physics is known.

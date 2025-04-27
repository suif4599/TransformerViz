# TransformerViz Manual

## Part 1: Understanding Transformers
### 1.1 What is a Transformer?
Imagine reading a novel:
- **Traditional approach**: Must read word by word (like early AI models such as RNNs)
- **Transformer approach**: Simultaneously sees the entire content and marks important connections with colored pens (this is the attention mechanism)

### 1.2 Core Components
#### Encoder (The Comprehender)
- Task: Deeply understand input text (e.g., a Chinese sentence)
- Operation: Repeatedly analyzes relationships between words (typically 6-12 layers)

#### Decoder (The Creator)
- Task: Generate new content based on understanding (e.g., translating to English)
- Special skill: Prevents peeking at future answers during generation (via look-ahead mask)

### 1.3 Attention Mechanism Explained
#### Basic Attention
When analyzing "The cat eats fish":
- At "eats", it focuses on both "cat" (who) and "fish" (what)
- Attention weights (0-100%) represent focus levels between words

#### Multi-Head Attention (Team Collaboration)
- Multiple attention "heads" analyze the same sentence simultaneously
- Each head focuses on different aspects (examples simplified for understanding):
  - Head 1: Analyzes "who does what to whom" (grammar)
  - Head 2: Identifies synonym replacement possibilities
  - Head 3: Captures emotional tendencies...

---

## Part 2: Key Concepts
### 2.1 Layers - Depth of Thought
- **First layer**: Identifies basic information (e.g., nouns/verbs)
- **Middle layers**: Builds phrase relationships (e.g., "black cat" as a unit)
- **Final layer**: Understands holistic meaning (infers "the cat might be hungry")

### 2.2 Attention Type Comparison
| Type                  | Function                      | Metaphor                     |
|-----------------------|-------------------------------|------------------------------|
| Encoder Self-Attention| Understands input relationships | Like referencing the full text during reading comprehension |
| Decoder Self-Attention| Ensures logical coherence      | Writing essays using only existing content |
| Encoder-Decoder Attention| Bridges input and output      | Translating while cross-referencing source text |

### 2.3 Why Look-Ahead Mask Matters
#### Without Mask:
- Generating "I love ___":
  - Model would directly fill "you" if it sees future answers during training
  - Equivalent to cheating on exams by pre-viewing answers

#### Mask Mechanism:
- When generating the 3rd word:
  - Allowed: [I, love, ? ]
  - Blocked: [I, love, you]
- Forces step-by-step reasoning

---

## Part 3: User Guide
### 3.1 Interface Map
- **Left Panel**: Model List → Choose different AI "brains"
- **Right Panel**:
  - Top: Console → Adjust observation modes
  - Middle: Input Box → Enter sentences to analyze
  - Bottom: Visualization → Click to reveal AI's "attention focus"

### 3.2 Console Details
#### AttentionPos (Observation Mode)
- **Encoder mode**: See how AI understands input
- **Decoder mode**: Watch word-by-word generation (with masking)
- **Encoder-Decoder mode**: Observe cross-language attention in translation/Q&A

#### LayerMixOption (Depth Control)
- **First**: Typical features (Recommended)
- **Final**: Abstract meanings (Not recommended - becomes human-unreadable in deep layers)
- **Average**: Layer synthesis (Not recommended)

#### HeadMixOption (Expert View)
- **All**: Simultaneous multi-expert view (Recommended)
- **First**: Single expert view (Less useful)
- **Average**: Expert consensus (Not recommended - loses information)

#### Temperature (Focus Intensity/Exploration)
- **Low (0.1)**: "Spotlight" effect - highlights key connections (low exploration)
- **High (5.0)**: "Floodlight" effect - reveals broad associations (high exploration)
- Warning: Avoid values greater than 1

---

## Troubleshooting Guide
- **No response after click** → Check: Input length? Try shorter sentences
- **Gray attention maps** → Solution: Lower Temperature value
- **Disabled model options** → Note: Model capabilities vary - try switching
- **Crashes/Errors** → Submit an issue for resolution

---

This manual balances technical accuracy with approachable metaphors to help users intuitively grasp Transformer operations. The temperature warning and layer recommendations reflect practical insights from model observation.
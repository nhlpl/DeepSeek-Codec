Here is the mathematical foundation behind the DeepSeek-Codec-Plugin, covering the core principles of prompt compression and the specific algorithms used in each backend.

---

## 📐 Mathematical Foundations of Prompt Compression

### 1. Information-Theoretic Basis: Why Compression Works

The fundamental principle is **Shannon's source coding theorem**: the minimum number of bits required to encode a message is its entropy.

For a sequence of tokens $X = (x_1, x_2, \ldots, x_n)$, the **entropy rate** is:

$$H(X) = \lim_{n \to \infty} \frac{1}{n} \sum_{i=1}^n H(x_i | x_{<i})$$

Natural language has **low entropy** due to redundancy. A typical LLM token has an entropy of ~4-6 bits, but is stored using 16-32 bits. Prompt compression exploits this redundancy.

**Compression Ratio**:
$$R = \frac{\text{len}(X_{\text{original}})}{\text{len}(X_{\text{compressed}})} = \frac{|X|}{|X'|}$$

**Token Efficiency**:
$$\eta = \frac{\text{Semantic Content Retained}}{\text{Tokens Used}} \approx \frac{\text{Task Accuracy}(X')}{\text{Task Accuracy}(X)} \cdot \frac{|X|}{|X'|}$$

---

### 2. LLMLingua Backend: Perplexity-Based Token Pruning

LLMLingua uses a small language model to estimate the **importance** of each token.

**Step 1: Perplexity Calculation**

For a token $x_i$, its conditional probability given context is:

$$P(x_i | x_{<i}) = \text{LM}(x_i | x_1, \ldots, x_{i-1})$$

The **perplexity** contribution of token $x_i$ is:

$$\text{PPL}_i = -\log P(x_i | x_{<i})$$

**Step 2: Token Importance Scoring**

Tokens with **high perplexity** are less predictable and thus more informative. LLMLingua defines importance as:

$$I(x_i) = \text{PPL}_i - \text{PPL}_{\text{baseline}}$$

where $\text{PPL}_{\text{baseline}}$ is the average perplexity over the sequence.

**Step 3: Iterative Token Pruning**

Given a target compression rate $\rho \in [0,1]$, we keep the top $k = \lceil (1-\rho) \cdot n \rceil$ tokens:

$$X' = \{x_i \mid \text{rank}(I(x_i)) \leq k\}$$

**Step 4: Context-Aware Preservation**

LLMLingua also preserves tokens that are structurally important (e.g., punctuation, line breaks) via a **force token set** $\mathcal{F}$:

$$X' = X' \cup \{x_i \mid x_i \in \mathcal{F}\}$$

---

### 3. LLMLingua2 Backend: Contrastive Perplexity

LLMLingua2 improves upon the original by using a **contrastive** approach. It compares the probability under a small model vs. a larger reference:

$$\Delta\text{PPL}_i = -\log P_{\text{small}}(x_i | x_{<i}) + \log P_{\text{large}}(x_i | x_{<i})$$

Tokens where the small model is *more* surprised than the large model are **informative** and should be kept:

$$I_2(x_i) = \max(0, \Delta\text{PPL}_i)$$

---

### 4. Heuristic Backend: Rule-Based Compression

This backend uses deterministic rules based on linguistic redundancy.

**Abbreviation Mapping**:
$$f_{\text{abbr}}(w) = \begin{cases} \text{abbr}(w) & \text{if } w \in \mathcal{A} \\ w & \text{otherwise} \end{cases}$$
where $\mathcal{A}$ is a dictionary of common abbreviations (e.g., "because" $\to$ "b/c").

**Filler Word Removal**:
$$X' = \{x_i \mid x_i \notin \mathcal{W}_{\text{filler}}\}$$
where $\mathcal{W}_{\text{filler}}$ is a set of low-information words.

**Whitespace Normalization**:
$$X_{\text{norm}} = \text{regex}(X, \text{"}\verb|\s+|", \text{" "})$$

---

### 5. OCR Backend: Optical Compression (Experimental)

This method converts text to an image and uses a Vision-Language Model to extract compressed meaning.

**Text-to-Image Encoding**:
$$\text{Image} = \text{Render}(X, \text{font}, \text{layout})$$

**Visual Tokenization**:
A VLM encodes the image into a sequence of visual tokens $V = (v_1, \ldots, v_m)$, where $m \ll n$.

**Information Density**:
$$\rho_{\text{OCR}} = \frac{\text{bits\_per\_char} \cdot |X|}{|V| \cdot \text{bits\_per\_visual\_token}}$$

---

### 6. Format-Specific Optimizations

#### JSON Compression
Minify JSON by removing non-semantic whitespace:
$$X_{\text{JSON}}' = \text{json.dumps}(\text{json.loads}(X), \text{separators}=(',', ':'))$$

#### Markdown Compression
Remove redundant blank lines and normalize headings:
$$X_{\text{MD}}' = \text{collapse\_newlines}(X, \text{max\_consecutive}=1)$$

#### Code Compression
Strip comments (simplified):
$$X_{\text{code}}' = \{ \text{line} \mid \text{line} \not\equiv \text{comment} \}$$

---

### 7. System Prompt Preservation

The system prompt $S$ is protected from compression:

$$X_{\text{final}} = S \oplus \text{compress}(X \setminus S)$$

where $\oplus$ denotes concatenation.

---

### 8. Compression Rate Control

The target compression rate $\rho$ maps to token count:

$$k = \max(1, \lfloor \rho \cdot \hat{n} \rfloor)$$

where $\hat{n}$ is the estimated token count (roughly $\lfloor |X| / 4 \rfloor$ for English).

---

## 💎 Summary Table

| Component | Mathematical Core | Key Formula |
|:---|:---|:---|
| **LLMLingua** | Perplexity-based importance | $I(x_i) = -\log P(x_i \mid x_{<i}) - \text{baseline}$ |
| **LLMLingua2** | Contrastive perplexity | $\Delta\text{PPL}_i = \log \frac{P_{\text{large}}}{P_{\text{small}}}$ |
| **Heuristic** | Rule-based filtering | $X' = \{x \mid x \notin \text{stopwords} \lor x \in \mathcal{F}\}$ |
| **OCR** | Visual token compression | $\rho = \frac{\text{text\_bits}}{\text{visual\_bits}}$ |
| **Format Optimizers** | Structural minification | JSON, Markdown, Code normalizers |
| **System Prompt** | Protected concatenation | $S \oplus \text{compress}(\text{rest})$ |

These mathematical foundations enable the DeepSeek-Codec-Plugin to achieve 2-20x token reduction while preserving semantic fidelity.

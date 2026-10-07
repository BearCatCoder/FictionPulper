# Tokenizer Comparison

Corpus SHA-256: `7cd6323095a33336e4a002d96554aebd9deb26b5a6971f3d2044e67dfb035f5b`
Split SHA-256: `01e708c7454a044539921bc179ff1b9b3f31aba616d90a1b22b6ea3e628a1229`

## Train

| Metric | Original | Tokenizer v2 |
|---|---:|---:|
| Total tokens | 8780719 | 8543829 |
| Tokens/word | 1.7078220011700929 | 1.6617476473663573 |
| Bytes/token | 3.2780402151577794 | 3.3689286150272904 |
| Minimum story tokens | 439 | 446 |
| Median story tokens | 3737 | 3732 |
| Mean story tokens | 5610.683067092652 | 5459.315654952076 |
| 95th percentile | 14594 | 13868 |
| Maximum story tokens | 21060 | 19208 |
| % <= 1024 | 1.7891373801916932 | 1.7891373801916932 |
| % <= 2048 | 16.549520766773163 | 16.61341853035144 |
| % <= 4096 | 53.9297124600639 | 54.37699680511182 |

## Validation

| Metric | Original | Tokenizer v2 |
|---|---:|---:|
| Total tokens | 567499 | 565661 |
| Tokens/word | 1.5807067094502751 | 1.5755871604608152 |
| Bytes/token | 3.423549645021401 | 3.4346737710395447 |
| Minimum story tokens | 511 | 513 |
| Median story tokens | 3714 | 3741 |
| Mean story tokens | 4082.726618705036 | 4069.5035971223024 |
| 95th percentile | 8468 | 8491 |
| Maximum story tokens | 10249 | 10286 |
| % <= 1024 | 2.8776978417266186 | 2.158273381294964 |
| % <= 2048 | 18.705035971223023 | 16.546762589928058 |
| % <= 4096 | 59.71223021582734 | 61.15107913669065 |

## Test

| Metric | Original | Tokenizer v2 |
|---|---:|---:|
| Total tokens | 558231 | 548897 |
| Tokens/word | 1.6937956270822334 | 1.6654742182332345 |
| Bytes/token | 3.271138650486985 | 3.3267644020645037 |
| Minimum story tokens | 881 | 890 |
| Median story tokens | 3328.5 | 3335.5 |
| Mean story tokens | 3987.364285714286 | 3920.692857142857 |
| 95th percentile | 9755 | 9475 |
| Maximum story tokens | 14841 | 14438 |
| % <= 1024 | 1.4285714285714286 | 1.4285714285714286 |
| % <= 2048 | 17.857142857142858 | 17.857142857142858 |
| % <= 4096 | 72.85714285714286 | 74.28571428571429 |

## Representative Passages

### Dialogue

Story: `f535b7f2963d` (MISS HENDERSON’S THANKSGIVING DAY)

> t was a quiet morning; and Miss Hetty’s thoughts kept time to the clicking of her knitting-needles.  “After all,” thought she, “it’s rather solitary taking dinner alone, and that on Thanksgiving Day. I remember, a long time ago, when my father

Original: 71 tokens; v2: 70 tokens.

### Names

Story: `3fad6bca32a7` (The Duchess Bredenbutta's Visit to Turvyland)

> The Duchess Bredenbutta was forty-seventh cousin to the Monarch of Mo and great-grandniece to the Queen; so you can readily see she was nearly related to the Princ

Original: 54 tokens; v2: 53 tokens.

### Contractions

Story: `f535b7f2963d` (MISS HENDERSON’S THANKSGIVING DAY)

> t the dignity of one of its own) came fully freighted, both inside and out. There were children and children’s children, who, in the pursuit of fortune, had strayed away from the homes where they first saw the light; but who were now returning, to re

Original: 66 tokens; v2: 66 tokens.

### Curly Quotation Marks

Story: `f535b7f2963d` (MISS HENDERSON’S THANKSGIVING DAY)

> f the other rooms,—which, though not furnished in so stately a manner, bear a family resemblance to “the best room,”—we will usher the reader into the opposite room, where he will find the owner and occupant of this prim-looking residence.

Original: 66 tokens; v2: 65 tokens.

### Em Dashes

Story: `f535b7f2963d` (MISS HENDERSON’S THANKSGIVING DAY)

> ust indeed be a poor household which, on this occasion, could not boast its turkey and plum-pudding,—those well-established dishes; not to mention its long rows of pies,—apple, mince, and pumpkin,—wherewith the Thanksgiving board is wont to

Original: 76 tokens; v2: 77 tokens.

### Historical Vocabulary

Story: `ec00c48af055` (PETER PLUNKETT’S ADVENTURE)

> lty of amusing mistakes. On one occasion, he addressed his housekeeper as “Most charming princess!” whereupon the good woman was led to entertain serious doubts as to his sanity, which, indeed, were not wholly unreasonable, since, though an excellen

Original: 65 tokens; v2: 67 tokens.

### Science Fiction Vocabulary

Story: `5c00c4892971` (The Bishop of Borglum and his Warriors)

> e from the window--a ship has come ashore. It has struck, and is fast embedded in the sand; but the rocket apparatus has thrown a rope on board, and formed a bridge from the wreck to the mainland; and all on board are saved, and reach the land, a

Original: 72 tokens; v2: 72 tokens.

### Western Terminology

Story: `e6092285d832` (BOYS)

> re very hot. He looked up at Katya once more and said:  "When a herd of bisons stampedes across the prairie the earth trembles, and the frightened mustangs kick and neigh."  He smiled impressively and added:  "And the Indians attack the trains, to

Original: 74 tokens; v2: 76 tokens.

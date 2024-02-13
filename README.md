# lichess-games-analysis

A project for learning some processes in data science and building data visualisations to provide insight!
The lichess dataset was selected because I like chess. 

Notes (for Kemp)

**Endgame insights** - seems hard

Maybe:
  - Classify games into different types of endgames - use chess rules to determine the pieces remaining on the board. For instance, if only kings and pawns are left, classify it as 'King and Pawn' endgame.
  - Separate games that ended in a win from those that were draws or losses.
      
**Dataset cannot be used for mistake analysis** - need data from a chess engine

**Piece Activity**
- **Defining Critical Squares**: Determine what constitutes a critical square. For pawn promotion, this would be the 8th rank for White and the 1st rank for Black.
- Then analyse when and how often pieces reach these critical squares.
    - For pawns:
        - Count how many times pawns are promoted in the dataset.
        - Examine the context of these promotions - for instance, in which types of positions or game phases (opening, middle game, endgame) they occur most frequently.
        - Look for any patterns in pawn promotion, such as specific opening moves leading to higher promotion rates.
- Heatmaps or bar charts - illustrate the frequency and distribution of piece movements to critical squares.
- Compare these frequencies across different levels of play or different player demographics?
- **Correlation with winning:**
    - **Creating a Correlation Variable**: Create a binary variable indicating the occurrence of the event (e.g., pawn promotion) in each game.
    - **Logistic Regression**: If analyzing a binary outcome (win/loss - what about draw?), to quantify the relationship between promotion and winning.
    - The correlation varies by the strength of the players (e.g., different correlations in amateur vs. professional games)?
    - What graph or chart to use?

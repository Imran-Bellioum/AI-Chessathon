AI Chessathon — Python Chess Engine
A chess engine I built for the 2026 AI Chessathon. I qualified for the London finals and finished in the top 10% globally.

I wanted to see how far I could get by combining algorithms, evaluation and careful testing, all under a strict time limit. The interesting part wasn't just finding a good move; it was making sure the engine could reliably find one quickly enough.
How the engine works
The main submission is agent.py. It takes a position in FEN format and returns a move in UCI notation.
- Search: iterative-deepening negamax with alpha-beta pruning / principal variation search (PVS)
- Tactical calculation: quiescence search to avoid stopping the search in the middle of an exchange
- Efficiency: transposition tables, move ordering using killer/history heuristics, and evaluation caching
- Position evaluation: material, piece-square tables and positional factors such as pawn structure and king safety
- Speed: NumPy and Numba for performance-sensitive evaluation code
- Reliability: time budgeting, a legal-move fallback, and handling of draws and repeated positions
- Endgames: optional Syzygy tablebase support when compatible files are available
Machine-learning experiment
As part of development, I also trained a logistic-regression position evaluator using positions from roughly 60,000 Chess.com and Lichess games, with Stockfish-informed analysis. It used piece-square features and additional positional features.

The aim was to see whether a learned evaluation could improve the decisions made by the search. I compared it with a simpler, hand-designed evaluator in head-to-head testing. The final submitted agent.py uses the classical evaluator; the learned model is disabled because the testing did not show a reliable enough improvement to justify keeping it.

That was one of the biggest lessons from the project: a more sophisticated model isn't necessarily a stronger engine.
Testing and iteration
I tested successive versions against a fixed opponent, recording match results as well as search depth, timing and reliability. I also checked low-time situations and edge cases rather than judging changes from a few promising positions. In one final 200-game benchmark, the retained version scored 48.25% against the reference champion.

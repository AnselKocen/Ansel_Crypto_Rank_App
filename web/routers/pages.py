"""
Page Routes - HTML page rendering for all tabs
"""

from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import FileResponse, RedirectResponse
from pathlib import Path

router = APIRouter(tags=["pages"])


@router.get("/ranked-portfolios")
def ranked_portfolio_page():
    report = Path(__file__).resolve().parents[2] / "artifacts/historical_replay/portfolio_evaluation/report/index.html"
    if report.is_file():
        return FileResponse(report, media_type="text/html")
    from fastapi.responses import HTMLResponse
    return HTMLResponse("<html lang='zh-CN'><meta charset='utf-8'><h1>排名选币组合回放</h1>"
                        "<p>组合报告尚未生成。</p></html>", status_code=202)


@router.get("/target-experiment")
def target_experiment_page():
    """Open the separate, paired clipped/raw training-target experiment."""
    report = Path(__file__).resolve().parents[2] / "artifacts/historical_replay/target_experiment/report/index.html"
    if report.is_file():
        return FileResponse(report, media_type="text/html")
    from fastapi.responses import HTMLResponse
    return HTMLResponse("<html lang='zh-CN'><meta charset='utf-8'><title>训练目标对照</title>"
                        "<h1>训练目标对照</h1><p>对照试验尚未完成。</p></html>", status_code=202)


@router.get("/historical-replay")
def historical_replay_page():
    """Open the offline-capable historical forecast comparison dashboard."""
    report = Path(__file__).resolve().parents[2] / "artifacts/historical_replay/report/index.html"
    if report.is_file():
        return FileResponse(report, media_type="text/html")
    from fastapi.responses import HTMLResponse
    return HTMLResponse(
        "<html lang='zh-CN'><meta charset='utf-8'><title>历史回放</title>"
        "<body style='background:#0b101b;color:#eef4ff;font:18px sans-serif;padding:48px'>"
        "<h1>历史预测回放</h1><p>数据准备或回放尚未完成，完成后本页将展示真实结果。</p>"
        "<a style='color:#63a5ff' href='/'>返回项目</a></body></html>",
        status_code=202,
    )

# Setup templates
BASE_DIR = Path(__file__).resolve().parent.parent.parent
templates = Jinja2Templates(directory=BASE_DIR / "web" / "templates")


def render_page(request, name, active_tab, **extra_context):
    # Starlette 1.x requires request first; retain compatibility with the older
    # runtime originally used by this project.
    import inspect
    context = {"request": request, "active_tab": active_tab, **extra_context}
    if "request" in inspect.signature(templates.TemplateResponse).parameters:
        return templates.TemplateResponse(request=request, name=name, context=context)
    return templates.TemplateResponse(name=name, context=context)



@router.get("/")
async def home_page(request: Request):
    """Home Page - Landing page with orbital animation"""
    return FileResponse(BASE_DIR / "web" / "templates" / "home.html")


@router.get("/educational")
async def educational_page(request: Request):
    """Tab 1: Educational Resources"""
    return render_page(request, "educational.html", "educational")


@router.get("/prediction")
async def prediction_page(request: Request):
    """Tab 2: Run Prediction"""
    return render_page(request, "prediction.html", "prediction")


@router.get("/sentiment")
async def sentiment_page(request: Request):
    """Tab 3: Sentiment Analysis"""
    return render_page(request, "sentiment.html", "sentiment")


@router.get("/ai-analysis", include_in_schema=False)
async def ai_analysis_page():
    """Temporarily hide the report page while retaining its backend and template."""
    return RedirectResponse(url="/", status_code=302)


@router.get("/ai-qa")
async def ai_qa_page(request: Request):
    """Tab 5: AI Q&A Assistant"""
    return render_page(request, "ai_qa.html", "ai-qa")


@router.get("/live-prices")
async def live_prices_page(request: Request):
    """Tab 6: Live Prices (TradingView)"""
    return render_page(request, "live_prices.html", "live-prices")


@router.get("/investment")
async def investment_page(request: Request):
    """Tab 7: Investment Recommendations"""
    return render_page(request, "investment.html", "investment")


@router.get("/game")
async def game_page(request: Request):
    """Tab: Crypto Cat Adventure Game"""
    import hashlib
    # One content version for the entire bundle prevents old cached controllers
    # from being combined with newly split state and scene files.
    digest = hashlib.sha256()
    for asset in ("css/game.css", "css/postal.css", "css/game-cosmic.css", "css/game-gallery.css", "js/game-state.js",
                  "js/postal-art.js", "js/postal-game.js", "css/castle.css",
                  "js/castle-art.js", "js/castle-game.js", "css/adventure.css",
                  "js/adventure-games.js", "js/adventure-worlds.js", "js/adventure-art.js", "js/game.js"):
        digest.update((BASE_DIR / "web" / "static" / asset).read_bytes())
    response = render_page(request, "game.html", "game",
                           game_asset_version=digest.hexdigest()[:16])
    response.headers["Cache-Control"] = "no-store"
    return response


@router.get("/transparency")
async def transparency_page(request: Request):
    """Tab 8: Transparency - Overview page for model details"""
    return render_page(request, "transparency.html", "transparency")


# Transparency Sub-pages
@router.get("/model-strategy")
async def model_strategy_page(request: Request):
    """Transparency Sub-page: Model Strategy"""
    return render_page(request, "model_strategy.html", "model-strategy")


@router.get("/hyper-params")
async def hyper_params_page(request: Request):
    """Transparency Sub-page: Hyperparameters"""
    return render_page(request, "hyper_params.html", "hyper-params")


@router.get("/features")
async def features_page(request: Request):
    """Transparency Sub-page: Feature Importance"""
    return render_page(request, "features.html", "features")


@router.get("/backtest")
async def backtest_page(request: Request):
    """Transparency Sub-page: Backtest Results"""
    return render_page(request, "backtest.html", "backtest")

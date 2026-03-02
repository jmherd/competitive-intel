# app.py
# Streamlit dashboard for the Competitive Intelligence System.
# Displays structured agent outputs in a professional multi-tab dashboard.

import streamlit as st
import plotly.graph_objects as go
import uuid
import json
from crew import run_intelligence_crew
from reports import initialize_database, save_report, get_all_reports, get_report_by_id, delete_report
from agents import CompanyProfile, CompetitorAnalysis, NewsAnalysis, IntelligenceReport

# ============================================================
# SHARED DISPLAY FUNCTION
# ============================================================
def display_results(results):
    """Renders the four-tab intelligence dashboard."""
    company = results.get("company")
    competitors = results.get("competitors")
    news = results.get("news")
    report = results.get("report")
    company_name = results.get("company_name", "Company")

    st.success(f"✅ Intelligence report complete for **{company_name}**")
    st.divider()

    tab1, tab2, tab3, tab4 = st.tabs([
        "🏢 Company Profile",
        "⚔️ Competitors",
        "📰 News & Sentiment",
        "📊 Intelligence Report"
    ])

    with tab1:
        if company:
            st.subheader(f"🏢 {company.name}")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Founded", company.founded)
            col2.metric("Headquarters", company.headquarters)
            col3.metric("Employees", company.employee_count)
            col4.metric("Est. Revenue", company.estimated_revenue)
            st.divider()

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Business Model")
                st.write(company.business_model)
                st.subheader("Market Position")
                st.write(company.market_position)
                st.subheader("Key Products & Services")
                for product in company.key_products:
                    st.markdown(f"• {product}")

            with col2:
                st.subheader("✅ Strengths")
                for strength in company.strengths:
                    st.markdown(f"• {strength}")
                st.subheader("⚠️ Weaknesses")
                for weakness in company.weaknesses:
                    st.markdown(f"• {weakness}")
        else:
            st.warning("Company profile data could not be parsed.")

    with tab2:
        if competitors:
            st.subheader("Competitive Landscape")
            st.write(competitors.competitive_landscape)
            st.divider()

            if competitors.competitors:
                cols = st.columns(len(competitors.competitors))
                for i, comp in enumerate(competitors.competitors):
                    with cols[i]:
                        st.subheader(comp.name)
                        st.caption(comp.market_position)
                        st.write(comp.business_model)
                        st.markdown("**Strengths:**")
                        for s in comp.strengths:
                            st.markdown(f"• {s}")
                        st.markdown("**Weaknesses:**")
                        for w in comp.weaknesses:
                            st.markdown(f"• {w}")

            st.divider()
            col1, col2 = st.columns(2)
            with col1:
                st.subheader(f"✅ {company_name} Advantages")
                for adv in competitors.target_company_advantages:
                    st.markdown(f"• {adv}")
            with col2:
                st.subheader(f"⚠️ {company_name} Disadvantages")
                for dis in competitors.target_company_disadvantages:
                    st.markdown(f"• {dis}")

            if competitors.competitors and company:
                st.divider()
                st.subheader("Competitive Positioning Chart")
                names = [company_name] + \
                    [c.name for c in competitors.competitors]
                strength_counts = [len(company.strengths)] + \
                    [len(c.strengths) for c in competitors.competitors]
                weakness_counts = [len(company.weaknesses)] + \
                    [len(c.weaknesses) for c in competitors.competitors]

                fig = go.Figure(data=[
                    go.Bar(name='Strengths Identified',
                           x=names, y=strength_counts,
                           marker_color='#2ecc71'),
                    go.Bar(name='Weaknesses Identified',
                           x=names, y=weakness_counts,
                           marker_color='#e74c3c')
                ])
                fig.update_layout(
                    barmode='group',
                    title="Strengths vs Weaknesses by Company",
                    xaxis_title="Company",
                    yaxis_title="Count",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Competitor analysis data could not be parsed.")

    with tab3:
        if news:
            col1, col2, col3 = st.columns(3)
            sentiment_color = {
                "positive": "🟢", "neutral": "🟡", "negative": "🔴"}
            col1.metric(
                "Overall Sentiment",
                f"{sentiment_color.get(news.overall_sentiment, '⚪')} "
                f"{news.overall_sentiment.title()}"
            )
            col2.metric("Sentiment Score", f"{news.sentiment_score}/10")
            col3.metric("News Items Found", len(news.news_items))
            st.divider()

            st.subheader("Key Themes")
            if news.key_themes:
                theme_cols = st.columns(len(news.key_themes))
                for i, theme in enumerate(news.key_themes):
                    theme_cols[i].info(theme)

            st.divider()
            st.subheader("Recent Developments")
            st.write(news.recent_developments)
            st.divider()

            st.subheader("News Items")
            for item in news.news_items:
                sentiment_emoji = {
                    "positive": "🟢", "neutral": "🟡",
                    "negative": "🔴"}.get(item.sentiment, "⚪")
                significance_badge = {
                    "high": "🔥 High", "medium": "📌 Medium",
                    "low": "📎 Low"}.get(item.significance, "📎 Low")
                with st.expander(
                    f"{sentiment_emoji} {item.headline} — {significance_badge}"
                ):
                    st.write(item.summary)
        else:
            st.warning("News analysis data could not be parsed.")

    with tab4:
        if report:
            outlook_colors = {
                "positive": "success", "neutral": "info", "negative": "warning"}
            outlook_emoji = {
                "positive": "📈", "neutral": "➡️", "negative": "📉"}
            getattr(st, outlook_colors.get(report.outlook, "info"))(
                f"{outlook_emoji.get(report.outlook, '➡️')} "
                f"Overall Outlook: **{report.outlook.title()}**"
            )
            st.divider()

            st.subheader("Executive Summary")
            st.write(report.executive_summary)
            st.divider()

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("🎯 Market Opportunity")
                st.write(report.market_opportunity)
                st.subheader("🛡️ Competitive Advantage")
                st.write(report.competitive_advantage)
            with col2:
                st.subheader("⚠️ Key Risks")
                for risk in report.key_risks:
                    st.markdown(f"• {risk}")
                st.subheader("💡 Strategic Recommendations")
                for i, rec in enumerate(report.strategic_recommendations, 1):
                    st.markdown(f"**{i}.** {rec}")
        else:
            st.warning("Intelligence report data could not be parsed.")

# Initialize database on startup
initialize_database()

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Competitive Intelligence",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 Competitive Intelligence System")
st.caption("Multi-agent AI research — company profiles, competitor analysis, news sentiment.")

# --- SESSION STATE ---
if "intel_results" not in st.session_state:
    st.session_state.intel_results = None
if "researching" not in st.session_state:
    st.session_state.researching = False

# --- MAIN TABS ---
main_tab1, main_tab2 = st.tabs(["🔍 New Research", "📚 Saved Reports"])


# ============================================================
# MAIN TAB 1 — NEW RESEARCH
# ============================================================
with main_tab1:

    st.divider()
    col1, col2 = st.columns([3, 1])

    with col1:
        company_name = st.text_input(
            "Company to research",
            placeholder="e.g. 'Tesla', 'Airbnb', 'Stripe'",
            disabled=st.session_state.researching
        )

    with col2:
        run_btn = st.button(
            "🚀 Run Intelligence",
            type="primary",
            use_container_width=True,
            disabled=st.session_state.researching or not company_name
        )

    if run_btn and company_name:
        st.session_state.researching = True
        st.session_state.intel_results = None

        status_box = st.empty()

        try:
            with st.spinner(
                f"Four agents researching {company_name}... this takes 2-3 minutes."
            ):
                def update_status(msg):
                    status_box.info(msg)

                results = run_intelligence_crew(
                    company_name=company_name,
                    status_callback=update_status
                )

            status_box.empty()

            # Save report to SQLite
            report_id = str(uuid.uuid4())
            save_report(report_id, company_name, results)

            st.session_state.intel_results = results
            st.session_state.researching = False
            st.rerun()

        except Exception as e:
            st.error(f"Research failed: {e}")
            st.session_state.researching = False

    # --- DISPLAY RESULTS ---
    if st.session_state.intel_results:
        display_results(st.session_state.intel_results)


# ============================================================
# MAIN TAB 2 — SAVED REPORTS
# ============================================================
with main_tab2:
    st.subheader("Saved Intelligence Reports")

    all_reports = get_all_reports()

    if not all_reports:
        st.info("No saved reports yet. Run your first research above.")
    else:
        st.metric("Total Reports", len(all_reports))
        st.divider()

        for saved in all_reports:
            with st.expander(
                f"🎯 {saved['company_name']} — {saved['created_at']}"
            ):
                # Parse and show executive summary if available
                if saved['report_data']:
                    try:
                        report_dict = json.loads(saved['report_data'])
                        st.write(report_dict.get('executive_summary', ''))

                        st.markdown("**Key Risks:**")
                        for risk in report_dict.get('key_risks', []):
                            st.markdown(f"• {risk}")

                        st.markdown("**Strategic Recommendations:**")
                        for rec in report_dict.get('strategic_recommendations', []):
                            st.markdown(f"• {rec}")

                        outlook = report_dict.get('outlook', 'neutral')
                        emoji = {"positive": "📈", "neutral": "➡️",
                                "negative": "📉"}.get(outlook, "➡️")
                        st.caption(f"{emoji} Outlook: {outlook.title()}")

                    except:
                        st.write("Report data available but could not be displayed.")

                if st.button("🗑️ Delete", key=f"del_{saved['id']}"):
                    delete_report(saved['id'])
                    st.success("Deleted.")
                    st.rerun()



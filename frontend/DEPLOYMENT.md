# Vercel demo

Set the Vercel root directory to `frontend`. This build displays the dashboard and labels the indicators and alerts as sample data. The deployment does not run Streamlit, connect to cameras, or execute YOLO inference.

The Windows launcher starts the Streamlit service separately. A live web deployment needs a persistent camera and inference service plus a frontend integration for camera discovery, video and detections.

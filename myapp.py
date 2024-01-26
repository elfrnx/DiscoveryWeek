import streamlit as st

st.title("Streamlit Cheat Sheet")
st.write("This is my first Streamlit app.")

# Header for the first section
st.header("Markdown")

# Write some markdown
st.write("## This is some Markdown text.")

# Header for the second section
st.header("Code")

# Write some code
st.code("""
for i in range(8):
  print("nlahtanla")
""")

# Header for the third section
st.header("Camera")

# Use the camera widget to take a picture
st.camera_input("Take a picture")


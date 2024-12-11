import React, { useState, useEffect } from "react";
import { LazyLoadImage } from "react-lazy-load-image-component";
import "./App.css";

const App = () => {
    const [styles, setStyles] = useState([]);
    const [selectedStyle, setSelectedStyle] = useState("");
    const [selectedFile, setSelectedFile] = useState(null);
    const [uploadedImage, setUploadedImage] = useState(null);
    const [outputImage, setOutputImage] = useState(null);
    const [isLoading, setIsLoading] = useState(false);
    const [isHovered, setIsHovered] = useState(false);

    const [style1, setStyle1] = useState("");
    const [style2, setStyle2] = useState("");
    const [blendWeight, setBlendWeight] = useState(0.5);
    const [isBlending, setIsBlending] = useState(false);
    const [blendedImage, setBlendedImage] = useState(null);


    useEffect(() => {
        const fetchStyles = async () => {
            try {
                const response = await fetch("http://localhost:5001/styles/list");
                const data = await response.json();
                setStyles(data);
                if (data.length > 0) {
                    setSelectedStyle(data[0].styleName);
                    setStyle1(data[0].styleName);
                    setStyle2(data[1]?.styleName || data[0].styleName);
                }
            } catch (error) {
                console.error("Error fetching styles:", error);
            }
        };
        fetchStyles();
        const slider = document.querySelector(".slider-container input[type='range']");
        if (slider) {
            const gradient = `linear-gradient(to right, #ff9800 0%, #ff9800 ${
                blendWeight * 100
            }%, #3f51b5 ${blendWeight * 100}%, #3f51b5 100%)`;
            slider.style.background = gradient;
        }
    }, [blendWeight]);

    const handleFileChange = (event) => {
        const file = event.target.files[0];
        setSelectedFile(file);
        setUploadedImage(URL.createObjectURL(file));
        setOutputImage(null);
        setBlendedImage(null);
    };

    const handleStyleSelect = (styleName) => {
        setSelectedStyle(styleName);
    };

    const handleApplyStyle = async () => {
        if (!selectedFile) {
            alert("Please upload an image!");
            return;
        }
        if (!selectedStyle) {
            alert("Please select a style!");
            return;
        }

        const formData = new FormData();
        formData.append("image", selectedFile);
        formData.append("style", selectedStyle);

        try {
            setIsLoading(true);
            const response = await fetch("http://localhost:5001/upload", {
                method: "POST",
                body: formData,
            });

            if (!response.ok) {
                throw new Error(`Failed to apply style: ${response.statusText}`);
            }

            const data = await response.json();
            if (data.imagePath) {
                setOutputImage(`http://localhost:5001${data.imagePath}`);
            } else {
                throw new Error("Invalid response from server.");
            }
        } catch (error) {
            console.error("Error applying style:", error);
            alert("Failed to apply style. Please try again.");
        } finally {
            setIsLoading(false);
        }
    };

    const handleDownload = async (imageUrl, fileName) => {
        if (!imageUrl) {
            alert("No image available to download.");
            return;
        }

        try {
            const response = await fetch(imageUrl);
            if (!response.ok) {
                throw new Error(`Failed to fetch image: ${response.statusText}`);
            }
            const blob = await response.blob();
            const link = document.createElement("a");
            link.href = URL.createObjectURL(blob);
            link.download = fileName;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        } catch (error) {
            console.error("Error downloading image:", error);
            alert("Failed to download the image.");
        }
    };

    const handleBlendWeightChange = (event) => {
        const weight = parseFloat(event.target.value);
        console.log(event.target.value);
        setBlendWeight(weight);

        // 获取滑块 DOM 元素并更新背景颜色
        const slider = event.target;
        const gradient = `linear-gradient(to right, #ff9800 0%, #ff9800 ${weight * 100}%, #3f51b5 ${weight * 100}%, #3f51b5 100%)`;
        slider.style.background = gradient;

        console.log(`Blend weight updated: ${weight * 100}% / ${(1 - weight) * 100}%`);
    };


    const handleApplyBlendingStyle = async () => {
        if (!selectedFile) {
            alert("Please upload an image!");
            return;
        }
        if (!style1 || !style2 || style1 === style2) {
            alert("Please select two different styles!");
            return;
        }

        const formData = new FormData();
        formData.append("image", selectedFile);
        formData.append("style1", style1);
        formData.append("style2", style2);
        formData.append("weight", blendWeight);

        try {
            setIsBlending(true);
            const response = await fetch("http://localhost:5001/blend", {
                method: "POST",
                body: formData,
            });

            if (!response.ok) {
                throw new Error(`Failed to blend styles: ${response.statusText}`);
            }

            const data = await response.json();
            if (data.imagePath) {
                setBlendedImage(`http://localhost:5001${data.imagePath}`);
            } else {
                throw new Error("Invalid response from server.");
            }
        } catch (error) {
            console.error("Error blending styles:", error);
            alert("Failed to blend styles. Please try again.");
        } finally {
            setIsBlending(false);
        }
    };

    return (
        <div className="App">
            <div className="app-container">
                <h1 className="app-title">🎨 Style Transfer Application</h1>

                <div className="upload-section">
                    <h5>Upload Your Image</h5>
                    <input type="file" onChange={handleFileChange}/>
                </div>

                <div className="image-display">
                    {uploadedImage && (
                        <div>
                            <h6>Uploaded Image:</h6>
                            <div className="image-container">
                                <img src={uploadedImage} alt="Uploaded Preview"/>
                            </div>
                        </div>

                    )}
                    {outputImage && (
                        <div>
                            <h6>Styled Image:</h6>
                            <div
                                className="output-image-wrapper"
                                onMouseEnter={() => setIsHovered(true)}
                                onMouseLeave={() => setIsHovered(false)}
                            >
                                <img src={outputImage} alt="Styled Output" />
                                {isHovered && (
                                    <div
                                        className="download-overlay"
                                        onClick={() => handleDownload(outputImage, "styled_image.jpg")}
                                    >
                                        <i className="fas fa-download"></i> Download
                                    </div>
                                )}
                            </div>
                        </div>
                    )}
                </div>

                <div className="style-section">
                    <h6>Select a Style</h6>
                    <div className="style-grid">
                        {styles.map(({styleName, thumbnailPath}) => (
                            <div
                                key={styleName}
                                className={`style-card ${
                                    selectedStyle === styleName ? "selected" : ""
                                }`}
                                onClick={() => handleStyleSelect(styleName)}
                            >
                                <LazyLoadImage
                                    src={`http://localhost:5001${thumbnailPath}`}
                                    alt={styleName}
                                    effect="blur"
                                />
                                <p>{styleName}</p>
                            </div>
                        ))}
                    </div>
                </div>

                <button
                    className="apply-btn"
                    onClick={handleApplyStyle}
                    disabled={isLoading}
                >
                    {isLoading ? "Applying Style..." : "Apply Style"}
                </button>

                <hr/>

                <div className="blending-section">
                    <h6>Blend Two Styles</h6>

                    <div className="style-grid">
                        {styles.map(({styleName, thumbnailPath}) => (
                            <div
                                key={styleName}
                                className={`style-card ${
                                    style1 === styleName ? "selected" : style2 === styleName ? "selected-secondary" : ""
                                }`}
                                onClick={() => {
                                    if (style2 === styleName) {
                                        setStyle2("");
                                    } else if (style1 === styleName) {
                                        setStyle1("");
                                    } else if (!style1) {
                                        setStyle1(styleName);
                                    } else if (!style2) {
                                        setStyle2(styleName);
                                    }
                                }}
                            >
                                <LazyLoadImage
                                    src={`http://localhost:5001${thumbnailPath}`}
                                    alt={styleName}
                                    effect="blur"
                                />
                                <p>{styleName}</p>
                            </div>
                        ))}
                    </div>

                    <div className="slider-container">
                        <input
                            type="range"
                            min="0"
                            max="1"
                            step="0.01"
                            value={blendWeight}
                            onChange={handleBlendWeightChange}
                        />
                        <p>
                            <span style={{color: "#ff9800"}}>
                                {Math.round(blendWeight * 100)}%
                            </span>{" "}
                                                /
                                                <span style={{color: "#3f51b5"}}>
                                {Math.round((1 - blendWeight) * 100)}%
                            </span>
                        </p>
                    </div>

                </div>

                <div className="output-iamge">
                    {blendedImage && (
                        <div>
                            <h6>Blended Image:</h6>
                            <div
                                className="output-image-wrapper"
                                onMouseEnter={() => setIsHovered(true)}
                                onMouseLeave={() => setIsHovered(false)}
                            >
                                <img src={blendedImage} alt="Blended Output"/>
                                {isHovered && (
                                    <div
                                        className="download-overlay"
                                        onClick={() => handleDownload(blendedImage, "blended_image.jpg")}
                                    >
                                        <i className="fas fa-download"></i> Download
                                    </div>
                                )}
                            </div>
                        </div>
                    )}
                </div>
                <div className="button">
                    <button
                        className="apply-btn"
                        onClick={handleApplyBlendingStyle}
                        disabled={isBlending}
                    >
                        {isBlending ? "Applying Blending Styles..." : "Apply Blending Styles"}
                    </button>
                </div>
            </div>
        </div>
    );
};

export default App;

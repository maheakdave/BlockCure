import {createContext, useState} from 'react';


interface user {
    loggedin:boolean;
    port:string;
    login:() => void;
    logout:() => void;
    setPort:(port:string)=>void;
}
interface ContextProviderProps {
    children: React.ReactNode;
}

export const AuthContext = createContext<user>({loggedin:false,port:"",login:()=>{},logout:()=>{},setPort:()=>{}});


export const AuthProvider:React.FC<ContextProviderProps> = ({children}:ContextProviderProps) =>{
    const [loggedin,setLoggedIn] = useState<boolean>(false);
    const [port,setPort] = useState<string>("")
    const login = () => {
        setLoggedIn(true);
    }
    const logout = ()=>{
        setLoggedIn(false);
    }
    return (
        
            <AuthContext.Provider value = {{loggedin,port,login,logout,setPort}}>
                {children}
            </AuthContext.Provider>
        
    )
}